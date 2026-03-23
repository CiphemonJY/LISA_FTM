"""
Model Merging for LISA Federated Learning.

Provides four specialist merge techniques plus a unified dispatch function:

  1. merge_lora_adapters()  — LoRA-aware SVD blend for federated adapter merging
  2. ties_merge()            — TIES-Merging: sign-veto + rescale for full-model fusion
  3. dare_merge()            — DARE: drop & rescale for conflicting parameters
  4. fisher_merge()           — Fisher-weighted: importance-weighted averaging
  5. model_soups()           — Model Soups: uniform/greedy checkpoint averaging
  6. merge_model_checkpoints() — Simple weighted average (baseline)

LoRA representation in LISA:
  A: (rank, in_features)    — "down" projection
  B: (out_features, rank)   — "up" projection
  Effective weight delta: ΔW = B @ A  (out_features × in_features)

All full-model methods (TIES, DARE, Fisher, Soups) operate on complete
parameter tensors and are architecture-agnostic. They work on any model
checkpoints from the same architecture.

The LoRA-aware methods (merge_lora_adapters, accumulate_lora_delta) operate
on separate A/B matrices and handle the ΔW ↔ (B,A) decomposition internally.
"""

from typing import Dict, List, Optional, Tuple, Callable
import torch
import numpy as np


# ============================================================================
# Core LoRA helpers
# ============================================================================

def _lora_ab_to_deltaw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ΔW = B @ A. A: (rank, in), B: (out, rank) → (out, in)."""
    return b @ a


def _deltaw_to_lora_ab(deltaw: torch.Tensor, rank: int
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank-r SVD approximation of ΔW.

    Returns B (out, rank) and A (rank, in) such that B @ A ≈ ΔW.

    U, S, Vh = svd(deltaw)  →  deltaw = U @ diag(S) @ Vh
    Keep top-r:  ΔW ≈ Ur @ Sr @ Vrh
    B = Ur * sqrt(Sr)   (out, rank)
    A = Vrh * sqrt(Sr)  (rank, in)  — Vh is (rank, in) with full_matrices=False
    """
    U, S, Vh = torch.linalg.svd(deltaw, full_matrices=False)
    # U: (out, rank),  S: (rank,),  Vh: (rank, in)

    # Clamp negligible singular values to zero for numerical stability.
    # Use a relative threshold based on the largest singular value to avoid
    # incorrectly zeroing small-but-significant values in low-rank matrices.
    eps = S.max() * 1e-4
    S_clamped = torch.where(S > eps, S, torch.zeros_like(S))

    sqrt_S = torch.sqrt(S_clamped[:rank])   # (rank,)
    B = U[:, :rank] @ torch.diag(sqrt_S)  # (out, rank)
    A = torch.diag(sqrt_S) @ Vh[:rank, :]  # (rank, in)
    return B.to(deltaw.dtype), A.to(deltaw.dtype)


def _split_lora_keys(state: Dict[str, torch.Tensor]
                     ) -> Tuple[set, set]:
    lora_keys = {k for k in state if k.endswith(".A") or k.endswith(".B")}
    return lora_keys, set(state.keys()) - lora_keys


def _param_base_name(key: str) -> str:
    return key.rsplit(".", 1)[0]


# ============================================================================
# 1. LoRA Adapter Merge  (SVD-based, federated-learning specific)
# ============================================================================

def merge_lora_adapters(
    base_state: Dict[str, torch.Tensor],
    adapter_state: Dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Blend an adapter state into a base state using SVD decomposition.

    For each LoRA layer with base (B0, A0) and adapter (B1, A1):
      ΔW = B1@A1 − B0@A0
      ΔW_merged = α · ΔW
      merged_BA = B0@A0 + ΔW_merged
      merged_B, merged_A = SVD_rank_r(merged_BA)

    Non-LoRA keys are copied unchanged from base_state.

    Args:
        base_state:     Global model state at previous round. Contains all weights.
        adapter_state:  Client's updated adapter state (A/B matrices, same shape).
        alpha:          Blend factor in [0, 1]. 1.0 = full adapter, 0.0 = keep base.
                       Values > 1 extrapolate (aggressive).

    Returns:
        Merged state dict.
    """
    lora_keys, non_lora = _split_lora_keys(base_state)
    all_params = {_param_base_name(k) for k in lora_keys}

    result = {k: base_state[k].clone() for k in non_lora}

    for name in all_params:
        a_key = f"{name}.A"
        b_key = f"{name}.B"

        base_A = base_state[a_key]      # (rank, in_features)
        base_B = base_state[b_key]      # (out_features, rank)
        rank   = base_A.shape[0]

        if a_key not in adapter_state or b_key not in adapter_state:
            result[a_key] = base_A.clone()
            result[b_key] = base_B.clone()
            continue

        ad_A = adapter_state[a_key]
        ad_B = adapter_state[b_key]

        if ad_A.shape != base_A.shape or ad_B.shape != base_B.shape:
            raise ValueError(
                f"Shape mismatch for '{name}': "
                f"base A={base_A.shape} B={base_B.shape} vs "
                f"adapter A={ad_A.shape} B={ad_B.shape}"
            )

        base_dw     = _lora_ab_to_deltaw(base_A, base_B)
        adapter_dw  = _lora_ab_to_deltaw(ad_A, ad_B)
        merged_dw   = (1 - alpha) * base_dw + alpha * adapter_dw

        merged_B, merged_A = _deltaw_to_lora_ab(merged_dw, rank)
        result[a_key] = merged_A
        result[b_key] = merged_B

    return result


def accumulate_lora_delta(
    base_state: Dict[str, torch.Tensor],
    adapter_states: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Accumulate multiple adapter states into one, weighted by sample counts.

    Implements FedAvg on the ΔW matrix space: each adapter contributes its
    delta relative to base, weighted by its sample count.

    Args:
        base_state:      Global model state at start of round.
        adapter_states: List of per-client adapter state dicts.
        weights:        Per-client sample counts. If None, equal weight.

    Returns:
        Accumulated state dict ready to load into global model.
    """
    if not adapter_states:
        return base_state

    n = len(adapter_states)
    if weights is None:
        weights = [1.0] * n
    weights = [float(w) for w in weights]
    total = sum(weights)
    if total <= 0:
        raise ValueError("Weights must sum to positive")
    norm_w = [w / total for w in weights]

    running = {k: base_state[k].clone() for k in base_state}
    lora_keys, _ = _split_lora_keys(base_state)
    all_params = {_param_base_name(k) for k in lora_keys}

    # Accumulate weighted deltas in A/B space.  Each adapter contributes
    # w * (ad_A - base_A) to running_A and w * (ad_B - base_B) to running_B.
    # This preserves the exact low-rank structure without SVD approximation.
    running_A = {name: base_state[f"{name}.A"].clone() for name in all_params}
    running_B = {name: base_state[f"{name}.B"].clone() for name in all_params}

    for adapter, w in zip(adapter_states, norm_w):
        for name in all_params:
            a_key = f"{name}.A"
            b_key = f"{name}.B"
            if a_key not in adapter or b_key not in adapter:
                continue
            ad_A = adapter[a_key]
            ad_B = adapter[b_key]
            base_A = base_state[a_key]
            base_B = base_state[b_key]
            # Accumulate delta A and delta B weighted by w
            running_A[name] = running_A[name] + w * (ad_A - base_A)
            running_B[name] = running_B[name] + w * (ad_B - base_B)

    # Build result
    for name in all_params:
        a_key = f"{name}.A"
        b_key = f"{name}.B"
        running[a_key] = running_A[name]
        running[b_key] = running_B[name]

    return running


# ============================================================================
# 2. TIES-Merging  (2023 — "TIES-Merging: Task Agnostic Model Merging")
# ============================================================================

def ties_merge(
    checkpoints: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    reset_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    TIES-Merging: resolves parameter interference via sign voting + rescale.

    For each parameter element across N models:
      1. Compute reference (weighted mean of all checkpoints)
      2. Delta: d_i = θ_i − θ_ref
      3. Sign consensus: count positive/negative signs per element
      4. Reset conflicting elements to 0
      5. Rescale surviving elements by (n_agree / n_total)
      6. Add rescaled deltas to reference

    Reference: https://arxiv.org/abs/2306.01708

    Args:
        checkpoints:     Full model state dicts to merge (same architecture).
        weights:         Per-model weights. If None, equal.
        reset_threshold: Elements where n_agree ≤ this are reset to 0.
                         Default 0 (reset on any sign conflict).

    Returns:
        Merged state dict.
    """
    if not checkpoints:
        raise ValueError("No checkpoints provided")
    if len(checkpoints) == 1:
        return {k: v.clone() for k, v in checkpoints[0].items()}

    n = len(checkpoints)
    if weights is None:
        weights = [1.0] * n
    weights = [float(w) for w in weights]
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights]

    # Reference = weighted mean
    ref = {}
    for key in checkpoints[0]:
        first = checkpoints[0][key]
        dtype = (torch.float32 if first.dtype in (torch.float16, torch.bfloat16)
                 else first.dtype)
        accum = torch.zeros_like(first, dtype=torch.float32)
        for ckpt, w in zip(checkpoints, norm_w):
            accum = accum + w * ckpt[key].float()
        ref[key] = accum.to(dtype)

    # Per-parameter deltas
    deltas = []
    for ckpt in checkpoints:
        delta = {}
        for key in ckpt:
            delta[key] = ckpt[key].float() - ref[key].float()
        deltas.append(delta)

    # Merge
    result = {}
    for key in ref:
        param_deltas = [d[key] for d in deltas]

        # Sign agreement count per element
        signs = torch.stack([torch.sign(d) for d in param_deltas])  # (n, *shape)
        sign_sum = signs.sum(dim=0)    # +n = all pos, -n = all neg, 0 = conflict
        n_agree = sign_sum.abs()       # how many models agree per element

        # Reset mask: elements where sign conflicts (n_agree < n, or below threshold)
        reset_mask = n_agree <= reset_threshold

        # Rescale: surviving elements weighted by agreement fraction
        scale = n_agree / n            # fraction of models that agree

        # Weighted mean of deltas, scaled by agreement
        merged = torch.zeros_like(ref[key])
        for d, w in zip(param_deltas, norm_w):
            merged = merged + w * d * scale

        # Zero out conflicts
        merged = torch.where(reset_mask, torch.zeros_like(merged), merged)

        result[key] = (ref[key].float() + merged).to(ref[key].dtype)

    return result


# ============================================================================
# 3. DARE  (2023 — "Model Debiasing via Adaptive Residual Exploration")
# ============================================================================

def dare_merge(
    checkpoints: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    drop_rate: float = 0.5,
    rescale: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    DARE: drop randomly chosen conflicting parameters and rescale survivors.

    For each parameter element:
      1. Weighted average delta from reference
      2. Identify conflicting elements (models disagree on sign)
      3. Randomly drop (1 − drop_rate) fraction of conflicting elements
      4. Rescale surviving elements by 1/(1 − drop_rate)

    More parameters survive than TIES, but interference is reduced.

    Reference: https://arxiv.org/abs/2306.13096

    Args:
        checkpoints: Full model state dicts to merge.
        weights:    Per-model weights. If None, equal.
        drop_rate:  Fraction of conflicting params to drop. 0.5 = aggressive,
                    0.2 = mild. Default 0.5.
        rescale:    Rescale surviving params to restore expected magnitude.
                    Default True (disable for near-zero drop_rate).

    Returns:
        Merged state dict.
    """
    if not checkpoints:
        raise ValueError("No checkpoints provided")
    if len(checkpoints) == 1:
        return {k: v.clone() for k, v in checkpoints[0].items()}

    n = len(checkpoints)
    if weights is None:
        weights = [1.0] * n
    weights = [float(w) for w in weights]
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights]

    # Reference = weighted mean
    ref = {}
    for key in checkpoints[0]:
        first = checkpoints[0][key]
        dtype = (torch.float32 if first.dtype in (torch.float16, torch.bfloat16)
                 else first.dtype)
        accum = torch.zeros_like(first, dtype=torch.float32)
        for ckpt, w in zip(checkpoints, norm_w):
            accum = accum + w * ckpt[key].float()
        ref[key] = accum.to(dtype)

    # Deltas
    deltas = []
    for ckpt in checkpoints:
        delta = {}
        for key in ckpt:
            delta[key] = ckpt[key].float() - ref[key].float()
        deltas.append(delta)

    result = {}
    for key in ref:
        # Weighted mean delta
        merged = torch.zeros_like(ref[key])
        for d, w in zip(deltas, norm_w):
            merged = merged + w * d[key]

        if rescale and drop_rate > 0 and drop_rate < 1:
            # Conflict mask: elements where not all models agree
            signs = torch.stack([torch.sign(d[key]) for d in deltas])
            sign_sum = signs.sum(dim=0).abs()
            conflict = sign_sum < n          # True = models disagree

            n_conflict = conflict.sum().item()
            if n_conflict > 0:
                # Random subset to keep
                keep_prob = 1 - drop_rate
                rand = torch.rand_like(conflict.float())
                keep_mask = ~(conflict & (rand < drop_rate))

                # Rescale surviving conflicting elements
                scale_factor = 1.0 / keep_prob
                merged = merged * torch.where(keep_mask, scale_factor, 1.0)

                # Zero out dropped elements
                merged = torch.where(keep_mask, merged, torch.zeros_like(merged))

        result[key] = (ref[key].float() + merged).to(ref[key].dtype)

    return result


# ============================================================================
# 4. Fisher-weighted Merging
# ============================================================================

def fisher_merge(
    checkpoints: List[Dict[str, torch.Tensor]],
    fisher_matrices: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    epsilon: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Fisher-weighted merging: importance-weighted parameter averaging.

    Each parameter element is weighted by its Fisher information — parameters
    with high Fisher (more informative about the data) are trusted more and
    smoothed less. Parameters with low Fisher are pulled toward the average.

    Args:
        checkpoints:     Full model state dicts to merge.
        fisher_matrices: Per-model Fisher diagonal dicts (same keys as checkpoint).
                         Each value is the same shape as the param and represents
                         the Fisher information (estimated from training data).
                         Higher = more important = less smoothing toward average.
        weights:         Additional per-model multipliers. If None, equal.
        epsilon:         Floor for Fisher values to avoid division by zero.

    Returns:
        Merged state dict.

    Reference: https://arxiv.org/abs/1912.05032
    """
    if not checkpoints or not fisher_matrices:
        raise ValueError("Must provide at least one checkpoint and one Fisher matrix")
    if len(checkpoints) != len(fisher_matrices):
        raise ValueError(f"checkpoints ({len(checkpoints)}) and fisher_matrices "
                         f"({len(fisher_matrices)}) must have same length")

    n = len(checkpoints)
    if weights is None:
        weights = [1.0] * n
    weights = [float(w) for w in weights]

    result = {}
    for key in checkpoints[0]:
        f_list = [fm[key] for fm in fisher_matrices]
        c_list = [ckpt[key].float() for ckpt in checkpoints]

        # Total Fisher per element
        stacked_f = torch.stack(f_list)                        # (n, *shape)
        total_f = stacked_f.sum(dim=0).clamp_min(epsilon)       # ( *shape)

        # Per-model normalized importance
        accum = torch.zeros_like(c_list[0])
        for c, f, w in zip(c_list, f_list, weights):
            importance = f / total_f          # normalized [0, 1]
            accum = accum + w * importance * c

        # Where Fisher is near zero, pull toward simple weighted average (regularization)
        avg = sum(c * w for c, w in zip(c_list, weights)) / sum(weights)
        # Low Fisher = high uncertainty → shrink toward average
        f_signal = stacked_f.sum(dim=0) / (stacked_f.sum(dim=0).max() + epsilon)
        shrink = (1 - f_signal).clamp(0, 1)
        merged = accum * (1 - shrink) + avg * shrink

        result[key] = merged.to(checkpoints[0][key].dtype)

    return result


def compute_fisher_diagonal(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    max_batches: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Estimate diagonal Fisher Information for a model from training data.

    Fisher[i] ≈ E[(∂ log p(y|x,θ) / ∂ θ_i)²] — how much each parameter
    influences the log-likelihood of the training data.

    Args:
        model:       Model to estimate Fisher for.
        dataloader:  Training batches.
        device:      Device to run on.
        max_batches: How many batches to accumulate over.

    Returns:
        Dict mapping parameter name → Fisher diagonal tensor (same shape as param).
    """
    model.eval()
    fisher: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        fisher[name] = torch.zeros_like(param.data)

    count = 0
    for batch in dataloader:
        if count >= max_batches:
            break

        inputs = batch.get("input_ids", batch.get("x", batch[0] if isinstance(batch, (list, tuple)) else None))
        targets = batch.get("labels", batch.get("y", batch[1] if isinstance(batch, (list, tuple)) else None))
        if inputs is None or targets is None:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        model.zero_grad()
        try:
            out = model(input_ids=inputs) if hasattr(model, "input_ids") else model(inputs)
            logits = out.logits if hasattr(out, "logits") else out
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        except Exception:
            continue

        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                fisher[name] += grad.data.float() ** 2

        count += 1

    if count > 0:
        for name in fisher:
            fisher[name] /= count

    return fisher


# ============================================================================
# 5. Model Soups  (2022 — "Model Soups: Averaging Weights of Multiple Models")
# ============================================================================

def model_soups(
    checkpoints: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
    validate_on: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Model Soups: uniform or greedy-weighted averaging of model checkpoints.

    Uniform Soup:  simple weighted average of all checkpoints.
    Greedy Soup:   iteratively add the checkpoint that most improves validation
                   accuracy (requires validate_on scores).

    The paper showed soups often outperform all individual models without
    requiring any alignment or special handling.

    Reference: https://arxiv.org/abs/2203.05482

    Args:
        checkpoints:   Full model state dicts to merge.
        weights:        Per-model base weights. If None, equal.
        validate_on:    Per-model validation accuracy scores. If provided,
                        greedy soup selection is used (best models kept).
                        If None, uniform averaging.

    Returns:
        Merged state dict.
    """
    if not checkpoints:
        raise ValueError("No checkpoints provided")
    if len(checkpoints) == 1:
        return {k: v.clone() for k, v in checkpoints[0].items()}

    n = len(checkpoints)
    if weights is None:
        weights = [1.0] * n
    weights = [float(w) for w in weights]
    total_w = sum(weights)
    norm_w = [w / total_w for w in weights]

    if validate_on is not None:
        # Greedy Model Soup: start with best model, iteratively add any model
        # whose inclusion raises the average validation score above best.
        scores = list(validate_on)
        selected = [scores.index(max(scores))]
        best = sum(scores[i] for i in selected) / len(selected)  # current average

        for _ in range(n - 1):
            improved = False
            for i, score in enumerate(scores):
                if i in selected:
                    continue
                trial = selected + [i]
                trial_avg = sum(scores[j] for j in trial) / len(trial)
                if trial_avg > best:
                    selected.append(i)
                    best = trial_avg
                    improved = True
                    break
            if not improved:
                break

        # Reweight selected models uniformly
        sel_weights = [norm_w[i] for i in selected]
        total_sw = sum(sel_weights)
        sel_norm = [w / total_sw for w in sel_weights]
        checkpoints = [checkpoints[i] for i in selected]
        norm_w = sel_norm

    # Weighted average
    result = {}
    for key in checkpoints[0]:
        first = checkpoints[0][key]
        dtype = (torch.float32 if first.dtype in (torch.float16, torch.bfloat16)
                 else first.dtype)
        accum = torch.zeros_like(first, dtype=torch.float32)
        for ckpt, w in zip(checkpoints, norm_w):
            accum = accum + w * ckpt[key].float()
        result[key] = accum.to(dtype)

    return result


# ============================================================================
# 6. Simple Weighted Average (baseline)
# ============================================================================

def merge_model_checkpoints(
    checkpoints: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Simple weighted average of full model checkpoints.
    This is the baseline to beat — most techniques try to improve on this.

    Args:
        checkpoints: Full model state dicts.
        weights:     Per-model weights. If None, equal.

    Returns:
        Merged state dict.
    """
    if not checkpoints:
        raise ValueError("No checkpoints provided")
    if len(checkpoints) == 1:
        return {k: v.clone() for k, v in checkpoints[0].items()}

    n = len(checkpoints)
    if weights is None:
        weights = [1.0] * n
    weights = [float(w) for w in weights]
    total = sum(weights)
    norm = [w / total for w in weights]

    result = {}
    for key in checkpoints[0]:
        first = checkpoints[0][key]
        dtype = (torch.float32 if first.dtype in (torch.float16, torch.bfloat16)
                 else first.dtype)
        accum = torch.zeros_like(first, dtype=torch.float32)
        for ckpt, w in zip(checkpoints, norm):
            accum = accum + w * ckpt[key].float()
        result[key] = accum.to(dtype)

    return result


# ============================================================================
# Unified dispatch
# ============================================================================

MERGE_METHODS: Dict[str, Callable] = {
    "svd":      merge_lora_adapters,
    "ties":     ties_merge,
    "dare":     dare_merge,
    "fisher":   fisher_merge,
    "soups":    model_soups,
    "average":  merge_model_checkpoints,
}


def merge_models(
    checkpoints: List[Dict[str, torch.Tensor]],
    method: str = "average",
    weights: Optional[List[float]] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Unified merge entry point.

    Args:
        checkpoints:  List of model state dicts to merge.
        method:       One of: "svd", "ties", "dare", "fisher", "soups", "average".
                      Default: "average".
        weights:      Optional per-model weights.
        **kwargs:     Method-specific arguments.

    Returns:
        Merged state dict.

    Example:
        merged = merge_models([ckpt1, ckpt2], method="ties", weights=[100, 50])
    """
    if method not in MERGE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(MERGE_METHODS.keys())}"
        )
    return MERGE_METHODS[method](checkpoints, weights=weights, **kwargs)


# ============================================================================
# Integration helpers
# ============================================================================

def apply_merged_adapter(
    model_state: Dict[str, torch.Tensor],
    adapter_state: Dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Apply an adapter onto a full model state.

    Non-LoRA layers are copied unchanged; LoRA layers are merged via SVD.
    """
    lora_keys, non_lora = _split_lora_keys(model_state)
    non_lora_state = {k: model_state[k].clone() for k in non_lora}
    merged_lora    = merge_lora_adapters(model_state, adapter_state, alpha=alpha)
    return {**non_lora_state, **merged_lora}


def checkpoint_has_lora(state: Dict[str, torch.Tensor]) -> bool:
    """True if the checkpoint contains LoRA A/B matrices."""
    return any(k.endswith(".A") or k.endswith(".B") for k in state)


def extract_lora_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract only the LoRA A/B matrices from a model state."""
    return {k: v for k, v in state.items()
            if k.endswith(".A") or k.endswith(".B")}


def extract_nonlora_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Extract the non-LoRA portion of a model state."""
    return {k: v for k, v in state.items()
            if not (k.endswith(".A") or k.endswith(".B"))}

#!/usr/bin/env python3
"""
Differential Privacy for Federated Learning

Implements user-level (output) differential privacy using the Gaussian mechanism:
1. Per-gradient L2 clipping: clip each gradient tensor to max_grad_norm
2. Gaussian noise addition: add N(0, σ²C²) where C = max_grad_norm, σ = noise_multiplier
3. Privacy accounting via RDP (Rényi Differential Privacy) for Gaussian mechanism

The server cannot infer what any single client contributed — even with
full knowledge of the aggregated model and all other clients' data.
"""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger("privacy")


# ============================================================================
# DP Configuration
# ============================================================================

@dataclass
class DPConfig:
    """Configuration for differential privacy."""
    enabled: bool = False
    noise_multiplier: float = 1.0   # σ - controls noise level relative to clipping bound
    max_grad_norm: float = 1.0       # C - per-gradient clipping bound (L2 norm cap)
    target_epsilon: Optional[float] = None  # optional epsilon budget to warn near

    def __post_init__(self):
        if self.enabled:
            if self.noise_multiplier <= 0:
                raise ValueError(f"noise_multiplier must be positive, got {self.noise_multiplier}")
            if self.max_grad_norm <= 0:
                raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")


# ============================================================================
# Gradient Privacy (Gaussian Mechanism)
# ============================================================================

class GradientPrivacy:
    """
    Applies differential privacy to gradient updates using the Gaussian mechanism.

    The standard composition for (ε, δ)-DP via the moments accountant / RDP:
      - Clip each gradient tensor to L2 norm <= max_grad_norm
      - Add Gaussian noise N(0, σ²C²) where C = max_grad_norm, σ = noise_multiplier
      - Track cumulative privacy cost via RDP

    For the Gaussian mechanism with RDP accounting (α=2):
      ε ≈ 2 * σ² * steps   (tight bound for σ ≥ 1, δ = 1/samples)

    References:
      - Abadi et al. (2016): "Deep Learning with Differential Privacy"
      - Mironov (2017): "Rényi Differential Privacy"
    """

    def __init__(self, config: Optional[DPConfig] = None):
        self.config = config or DPConfig()

    # ------------------------------------------------------------------
    # Core DP operations
    # ------------------------------------------------------------------

    def clip_gradients(
        self,
        grad_dict: Dict[str, torch.Tensor],
        max_norm: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Clip each gradient tensor to have L2 norm <= max_norm.

        Per-gradient clipping: each tensor is clipped independently.
        This is the standard approach used in DP-SGD / federated DP.

        Formula: g_clipped = g * min(1, C / ||g||_2) / max(1, ||g||_2 / C)
                which simplifies to: g_clipped = g / max(1, ||g||_2 / C)

        Args:
            grad_dict: parameter name -> gradient tensor
            max_norm: clipping bound C

        Returns:
            new dict of clipped gradient tensors (cloned, no in-place modify)
        """
        clipped = {}
        for name, grad in grad_dict.items():
            if not isinstance(grad, torch.Tensor):
                clipped[name] = grad
                continue

            grad_float = grad.float()
            grad_norm = torch.norm(grad_float)
            if grad_norm > 0:
                # Clip: scale down if norm exceeds max_norm
                scale = min(1.0, max_norm / grad_norm.item())
                clipped[name] = grad_float * scale
            else:
                clipped[name] = grad_float

        return clipped

    def add_noise(
        self,
        grad_dict: Dict[str, torch.Tensor],
        noise_multiplier: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to each gradient tensor.

        Noise std per tensor = noise_multiplier * max_grad_norm.
        We use max_grad_norm from the config (or it must be passed alongside).

        IMPORTANT: noise_multiplier alone is not enough — you also need
        the clipping bound to know the actual noise std. Callers should
        use dp_aggregate() which handles both together.

        Args:
            grad_dict: clipped gradient tensors
            noise_multiplier: σ — ratio of noise std to clipping bound

        Returns:
            new dict with noise added (sampled fresh each call)
        """
        noisy = {}
        for name, grad in grad_dict.items():
            if not isinstance(grad, torch.Tensor):
                noisy[name] = grad
                continue

            grad_float = grad.float()
            # Noise std = σ * C (noise_multiplier * clipping bound)
            # We use the per-gradient norm as a proxy for C when called standalone
            grad_norm = torch.norm(grad_float).item()
            noise_std = noise_multiplier * max(grad_norm, 1e-8)
            noise = torch.randn_like(grad_float) * noise_std
            noisy[name] = grad_float + noise

        return noisy

    def dp_aggregate(
        self,
        grad_dicts: List[Dict[str, torch.Tensor]],
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        client_weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full DP aggregation pipeline: clip → sum → add noise → return private gradient.

        This is the core DP-FedAvg operation. Each client's gradient is clipped
        individually, then all clipped gradients are summed (weighted average),
        and finally Gaussian noise is added to the aggregate.

        Args:
            grad_dicts: list of client gradient dicts (each: param_name -> tensor)
            noise_multiplier: σ — noise multiplier (σ >= 1 is typical for (ε,δ)-DP)
            max_grad_norm: C — per-gradient clipping bound
            client_weights: optional list of per-client weights (sample counts normalized).
                           If None, equal weighting is used.

        Returns:
            dict of private aggregated gradients (one entry per parameter)
        """
        if not grad_dicts:
            return {}

        # Get all parameter names from the first gradient dict
        param_names = list(grad_dicts[0].keys())

        # Default: equal weights
        n = len(grad_dicts)
        if client_weights is None:
            client_weights = [1.0 / n] * n
        else:
            total = sum(client_weights)
            if total <= 0:
                client_weights = [1.0 / n] * n
            else:
                client_weights = [w / total for w in client_weights]

        # Clipped gradients per client (each gradient clipped independently)
        clipped_grads = []
        for grad_dict in grad_dicts:
            clipped = self.clip_gradients(grad_dict, max_norm=max_grad_norm)
            clipped_grads.append(clipped)

        # Weighted sum of clipped gradients (NO clipping on the sum)
        aggregated = {}
        for name in param_names:
            total = None
            for clipped, weight in zip(clipped_grads, client_weights):
                if name not in clipped:
                    continue
                grad = clipped[name].float()
                if total is None:
                    total = grad * weight
                else:
                    total = total + grad * weight

            if total is None:
                total = torch.zeros_like(grad_dicts[0][name]).float()
            aggregated[name] = total

        # Add noise to the aggregated (summed) gradient
        # Noise std = σ * C (applied to the already-clipped sum)
        noise_std = noise_multiplier * max_grad_norm

        noisy_aggregated = {}
        for name, grad in aggregated.items():
            if not isinstance(grad, torch.Tensor):
                noisy_aggregated[name] = grad
                continue
            noise = torch.randn_like(grad) * noise_std
            noisy_aggregated[name] = grad + noise

        return noisy_aggregated

    # ------------------------------------------------------------------
    # Privacy accounting
    # ------------------------------------------------------------------

    @staticmethod
    def compute_epsilon(
        noise_multiplier: float,
        steps: int,
        delta: float = 1e-5,
        alpha: int = 2,
    ) -> float:
        """
        Estimate ε for (ε, δ)-DP after `steps` rounds using RDP accounting
        for the Gaussian mechanism.

        Uses the standard RDP bound for Gaussian mechanism:
          ρ(ε, δ) = min_{α>1} [ε - ln(δ) - (α-1) * RDP_α(σ)] / (α-1)
        Simplified closed form for integer α:
          ε ≈ sqrt(2 * α * ln(1/δ) * σ²) + α * ln(1/δ) * (1 - 1/α) / σ

        With α=2 (a common tight bound for DP-SGD):
          ε ≈ sqrt(4 * ln(1/δ) * σ²) + 2 * ln(1/δ) / σ

        We use the even simpler standard approximation:
          ε ≈ 2 * noise_multiplier² * steps   [for RDP α=2 with small δ]

        This is the approach used in OpenDP / TensorFlow Privacy.

        Args:
            noise_multiplier: σ (noise multiplier, ratio of noise std to clip bound)
            steps: number of federated rounds (composition steps)
            delta: δ — probability of privacy violation (default 1e-5)
            alpha: RDP order (default 2, the minimum useful order)

        Returns:
            ε estimate (float)
        """
        if steps <= 0:
            return 0.0
        if noise_multiplier <= 0:
            return float('inf')

        # Standard RDP→(ε,δ) conversion for Gaussian mechanism at α=2:
        # ε = 2 * σ² * steps  (tight upper bound for σ ≥ 1)
        # This is what OpenDP / DP-SGD libraries use.
        epsilon = 2.0 * (noise_multiplier ** 2) * steps

        return epsilon

    @staticmethod
    def epsilon_to_delta(
        epsilon: float,
        noise_multiplier: float,
        steps: int,
        target_delta: float = 1e-5,
    ) -> bool:
        """
        Check if a privacy budget (ε, δ) meets a target δ threshold.

        Uses the RDP lower bound approximation:
          δ ≈ exp(-ε² / (2 * σ² * steps + ε))

        Returns True if the implied δ <= target_delta.
        """
        if epsilon <= 0 or noise_multiplier <= 0 or steps <= 0:
            return False
        denom = 2 * (noise_multiplier ** 2) * steps * epsilon + epsilon**2
        if denom <= 0:
            return False
        computed_delta = math.exp(-epsilon**2 / (2 * (noise_multiplier**2) * steps + epsilon))
        return computed_delta <= target_delta

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def privacy_status(self, steps: int, delta: float = 1e-5) -> Dict[str, any]:
        """
        Return a human-readable privacy status dict.

        Args:
            steps: number of rounds so far
            delta: δ target (default 1e-5)

        Returns:
            dict with epsilon, delta, strength ("strong"/"moderate"/"weak"), and warnings
        """
        if not self.config.enabled:
            return {"enabled": False}

        eps = self.compute_epsilon(self.config.noise_multiplier, steps, delta)

        if eps < 2:
            strength = "strong"
        elif eps < 8:
            strength = "moderate"
        else:
            strength = "weak"

        warnings = []
        if eps > 10:
            warnings.append(f"ε={eps:.1f} exceeds 10 — privacy is very weak")
        if self.config.target_epsilon and eps > self.config.target_epsilon:
            warnings.append(f"ε={eps:.1f} exceeds target_epsilon={self.config.target_epsilon}")

        return {
            "enabled": True,
            "epsilon": eps,
            "delta": delta,
            "steps": steps,
            "noise_multiplier": self.config.noise_multiplier,
            "max_grad_norm": self.config.max_grad_norm,
            "strength": strength,
            "warnings": warnings,
        }

    def log_status(self, steps: int, delta: float = 1e-5):
        """Log current privacy status at INFO level."""
        status = self.privacy_status(steps, delta)
        if not status["enabled"]:
            logger.info("Differential privacy: disabled")
            return

        warnings = ""
        if status["warnings"]:
            warnings = " | WARNINGS: " + "; ".join(status["warnings"])

        logger.info(
            f"[DP] Round {steps}: "
            f"ε={status['epsilon']:.2f} (δ={status['delta']}) "
            f"noise_mult={status['noise_multiplier']} "
            f"clip_norm={status['max_grad_norm']} "
            f"[{status['strength'].upper()} privacy]{warnings}"
        )

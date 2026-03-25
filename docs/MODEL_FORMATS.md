# Model Architecture Formats & LoRA Compatibility

This guide documents the different layer types used by major model families so the LoRA tool can be configured per-model.

---

## Model Family Quick Reference

| Model Family | Example Models | Layer Type | LoRA Pattern | Target Modules |
|-------------|---------------|-----------|-------------|---------------|
| **GPT-2 / DistilGPT2** | `distilgpt2`, `gpt2` | `transformers.pytorch_utils.Conv1D` | **AFTER** (output) | `attn.c_proj`, `attn.c_attn`, `mlp.c_proj`, `mlp.c_fc` |
| **Qwen2** | `Qwen2.5-0.5B`, `Qwen2.5-3B` | `torch.nn.Linear` | **BEFORE** (input) | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Llama** | `meta-llama/Llama-2-*` | `torch.nn.Linear` | **BEFORE** (input) | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Pythia / NeoX** | `pythia-160m`, `EleutherAI/pythia-*` | `torch.nn.Linear` OR `Conv1D` | **MIXED** | `query_key_value`, `dense`, `mlp`, `c_proj`, `c_attn` |
| **Phi** | `microsoft/phi-2`, `microsoft/phi-3-*` | `torch.nn.Linear` | **BEFORE** (input) | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| **Mistral** | `mistralai/Mistral-*` | `torch.nn.Linear` | **BEFORE** (input) | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

---

## Layer Type Details

### 1. `torch.nn.Linear` (Standard)
Used by: Qwen, Llama, Phi, Mistral, most modern transformers

**Characteristics:**
- Weight shape: `(out_features, in_features)`
- Forward: `y = x @ W.T + b`
- Input: `(..., in_features)` → Output: `(..., out_features)`

**LoRA Pattern: BEFORE (input)**
```
lora = x @ A.T @ B.T   # (..., in) @ (in, rank) @ (rank, out) = (..., out)
result = original(x) + lora * scaling
```
- `lora_A`: `(rank, in_features)`
- `lora_B`: `(out_features, rank)`

**Target module names:** `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention), `gate_proj`, `up_proj`, `down_proj` (MLP)

---

### 2. `transformers.pytorch_utils.Conv1D` (GPT-2 Style)
Used by: GPT-2, DistilGPT2, some GPT-NeoX variants

**Characteristics:**
- Weight shape: `(nf, nx)` — **TRANSPOSED** compared to Linear
  - `nx` = input features (e.g., 768 for GPT-2 small)
  - `nf` = output features (e.g., 768 for attention, 3072 for MLP intermediate)
- Forward: `y = x @ weight.T + bias` where weight is `(nf, nx)`
- Input: `(..., nx)` → Output: `(..., nf)` where **nf ≠ nx is possible**

**LoRA Pattern: AFTER (output)**
```
original = layer(x)              # (..., nx) → (..., nf)
lora = original @ A.T @ B.T     # (..., nf) @ (nf, rank) @ (rank, nf) = (..., nf)
result = original + lora * scaling
```
- `lora_A`: `(rank, nf)` — project from output dimension
- `lora_B`: `(nf, rank)` — project back to output dimension

**IMPORTANT:** Conv1D uses **LoRA_AFTER** because:
- The layer's input and output dimensions differ (nx ≠ nf)
- LoRA must add to the output, not the input
- A and B are sized for the OUTPUT dimension (nf), not input (nx)

**Target module names:** `attn.c_proj`, `attn.c_attn`, `mlp.c_proj`, `mlp.c_fc`

**Conv1D detection:**
```python
is_conv1d = isinstance(module, (nn.Conv1d, nn.modules.conv.Conv1d)) or type(module).__name__ == 'Conv1D'
# Check: hasattr(module, 'nf') and hasattr(module, 'nx')
# Conv1D nf/nx vs Linear in_features/out_features
```

---

### 3. Mixed Models (Some GPT-NeoX)
Used by: Some EleutherAI models, older OLMo variants

**Characteristics:**
- Some attention layers use `Linear`, others use `Conv1D`
- MLP layers may use either

**LoRA Pattern: DETECT PER-LAYER**
```python
for full_name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv1d)) or type(module).__name__ == 'Conv1D':
        # Check which type and apply appropriate LoRA
        if type(module).__name__ == 'Conv1D':
            # LoRA_AFTER
        else:
            # LoRA_BEFORE
```

**Target module names:** `query_key_value`, `dense`, `mlp`, `c_proj`, `c_attn`, `q_proj`, `k_proj`, `v_proj`, `o_proj`

---

## LoRA Application Patterns

### Pattern A: LoRA_BEFORE (Standard for Linear)
```python
def forward(self, x):
    # Apply LoRA to input, add to original output
    lora = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
    return self.linear(x) + lora * self.scaling
```
- A shape: `(rank, in_features)`
- B shape: `(out_features, rank)`
- Works for: Linear layers where input dim = LoRA's first dim

### Pattern B: LoRA_AFTER (For Conv1D or when input≠output)
```python
def forward(self, x):
    # Apply LoRA to output, add to original output
    original = self.linear(x)
    lora = original @ self.lora_A.T @ self.lora_B.T
    return original + lora * self.scaling
```
- A shape: `(rank, out_features)` 
- B shape: `(out_features, rank)`
- Works for: Conv1D where input(nx) ≠ output(nf), or any layer where you want to modify output

---

## Configuration Schema

To make this configurable per model:

```yaml
models:
  Qwen/Qwen2.5-0.5B:
    lora_pattern: "before"        # LoRA_BEFORE
    layer_type: "Linear"         # torch.nn.Linear
    target_modules:
      - "q_proj"
      - "k_proj" 
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"

  distilgpt2:
    lora_pattern: "after"        # LoRA_AFTER
    layer_type: "Conv1D"         # transformers.pytorch_utils.Conv1D
    target_modules:
      - "attn.c_proj"
      - "attn.c_attn"
      - "mlp.c_proj"
      - "mlp.c_fc"

  microsoft/phi-2:
    lora_pattern: "before"
    layer_type: "Linear"
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "dense"
```

---

## Auto-Detection Logic

```python
def detect_lora_pattern(module, target_name):
    """Auto-detect whether to use LoRA_BEFORE or LoRA_AFTER."""
    # Conv1D always needs AFTER
    if type(module).__name__ == 'Conv1D':
        return "after"
    
    # Standard Linear usually uses BEFORE
    if isinstance(module, nn.Linear):
        # Check if this is an attention projection layer
        if any(tm in target_name for tm in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            return "before"
        # MLP layers in modern models
        if any(tm in target_name for tm in ['gate_proj', 'up_proj', 'down_proj']):
            return "before"
        # Fallback
        return "before"
    
    return "before"  # default
```

---

## Weight Shape Reference

| Model | Layer | Weight Shape | nx (in) | nf (out) |
|-------|-------|-------------|---------|---------|
| distilgpt2 | attn.c_attn | (768, 2304) | 768 | 2304 |
| distilgpt2 | attn.c_proj | (768, 768) | 768 | 768 |
| distilgpt2 | mlp.c_fc | (3072, 768) | 768 | 3072 |
| distilgpt2 | mlp.c_proj | (768, 3072) | 3072 | 768 |
| Qwen2.5-0.5B | q_proj | (896, 896) | 896 | 896 |
| Qwen2.5-0.5B | o_proj | (896, 896) | 896 | 896 |
| Qwen2.5-0.5B | gate_proj | (896, 896) | 896 | 896 |
| Qwen2.5-0.5B | up_proj | (3584, 896) | 896 | 3584 |

---

*Last updated: 2026-03-24 — Tested with distilgpt2 (Conv1D) and Qwen2.5-0.5B (Linear)*

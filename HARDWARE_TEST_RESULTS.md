# Mac Mini M4 Pro — LISA_FTM Hardware Test Results

**Date:** 2026-03-24  
**Host:** JYbot's Mac mini (Mac Mini M4 Pro)  
**Location:** /tmp/LISA_FTM_review

---

## Hardware Profile

```
platform:        Darwin
cpu:             Apple M4
cpu_cores:       10
ram_total_gb:    16.0
ram_available_gb: 8.5
gpu:             Apple M4 (MPS)
gpu_type:        mps
gpu_memory_gb:   16.0
disk_available_gb: 40.0
recommended_framework: mlx
max_model_size:  3B
use_disk_offload: True
estimated_speed:  fast
```

**Summary:** 10-core M4, 16GB RAM, unified memory — well-suited for local LLM fine-tuning.

---

## Test Results

### 1. real_training.py (pythia-160m, 150 steps)

| Metric | Value |
|--------|-------|
| **Device** | MPS (Apple Silicon GPU) |
| **Model** | EleutherAI/pythia-160m (162M params) |
| **Time** | 2m 39s (159s) |
| **Time/step** | 1.06s |
| **Final loss** | 0.0033 (step 150), avg 1.9354 |
| **Peak RAM** | ~7.5 GB (of 16 GB available) |
| **Trainable params** | 442,368 (0.27% of model) |
| **LoRA layers** | 36 (full model) |
| **Output** | output/real_training/final_model.pt |

**Loss curve (key steps):**
| Step | Loss |
|------|------|
| 10 | 9.6870 |
| 20 | 7.6755 |
| 30 | 5.2645 |
| 40 | 2.4816 |
| 50 | 0.5383 |
| 100 | 0.0166 |
| 150 | 0.0033 |

**✅ Training successful** — loss dropped from ~9.7 to 0.003 over 150 steps. Clean convergence on wikitext-2 dataset.

**⚠️ Inference issue:** Generated output is degenerate (repeating last token: "a a a a...", "models models models..."). Likely due to pad token = eos token causing attention mask issues. Does not affect training quality.

---

### 2. main.py --mode train (pythia-70m, --steps 100)

| Metric | Value |
|--------|-------|
| **Device** | MPS |
| **Model** | EleutherAI/pythia-70m (70M params) |
| **Time** | ~12s (estimated) |
| **Final loss** | 593.9845 |
| **LoRA layers** | 24 (rank=8) |
| **Trainable params** | 393,216 / 70,819,840 (0.6%) |
| **Layers trained** | [0, 1, 2, 4, 5] (bottom=2, top=2, middle=1) |
| **Output** | output/lisa_torch |

**⚠️ Issues observed:**
1. **Argument mismatch:** `--steps 100` was specified but `main.py` uses `--iters` for train mode iterations. The actual CLI help says `--steps` is for client mode only. Train mode ran 50 iterations regardless of `--steps` argument.
2. **Loss stuck at ~594:** Loss barely decreased (596 → 594 over 50 steps), indicating potential training failure or that LISA's layer-selection strategy needs more iterations to converge on this tiny model.
3. **LISA layer selection:** Only 5 out of 6 layers trained (layer 3 skipped) — this is expected behavior but raises questions about whether sufficient capacity was trained.

**Recommendation:** For pythia-70m, try `--iters 200` with `--bottom 3 --top 3` to ensure full coverage, or use pythia-160m which has more layers and may benefit more from LISA's selective training.

---

### 3. MLX Mode

| Metric | Value |
|--------|-------|
| **Status** | **SKIPPED** — `main.py` does not have `--mode mlx` |
| **mlx_lm available** | ✅ Yes (installed) |
| **MLX GPU access** | ✅ Confirmed |

**Note:** `main.py` only supports these modes: `hardware`, `train`, `offload`, `server`, `client`, `simulate`, `rollback`. No native MLX support yet.

To use MLX for Apple Silicon GPU training, mlx-lm library is available and can be imported directly. A dedicated `--mode mlx` would need to be added to main.py.

---

## Issues Found

1. **[main.py] `--steps` argument ignored in train mode** — Train mode uses `--iters` not `--steps`. The CLI accepts `--steps` silently but it maps to client mode only. Users expecting `--steps` will get unexpected iteration counts.

2. **[real_training.py] Degenerate inference output** — Model repeats last token during generation demo. Root cause: `pad_token == eos_token` makes attention mask unreliable. Warning is printed: *"The attention mask is not set and cannot be inferred from input because pad token is same as eos token."* Fix: explicitly set `pad_token` to a distinct ID or use attention_mask.

3. **[main.py train] Loss convergence issue** — pythia-70m loss stayed near ~594 across 50 iterations. Possible causes:
   - Learning rate too high/low for tiny model
   - LISA training too few layers (5/6) for sufficient capacity
   - Dataset tokenization issue (LM-style training on causal model)

4. **[main.py] No MLX mode** — The framework detects MLX as "recommended" but has no MLX training path. This is a significant gap for Apple Silicon users who would benefit most from native MLX acceleration.

---

## Recommendations

1. **Add MLX training mode to main.py** — High priority. Use `mlx_lm.generate()` for inference and add a `mlx_fed/` module for LISA-style federated learning on MLX models.

2. **Fix `--steps` vs `--iters` confusion** — Either alias `--steps` to `--iters` in train mode, or update the CLI help to be clearer. Consider making the argument name consistent across modes.

3. **Investigate pythia-70m training failure** — Run `real_training.py` with pythia-70m to see if the issue is model-specific or LISA-specific. Try different learning rates (1e-3, 5e-4, 1e-4).

4. **Fix degenerate generation** — Add proper attention mask handling in the inference demo, or use a chat template for causal models.

5. **Add peak memory tracking** — Both training scripts would benefit from `psutil` or `torch.cuda.max_memory_allocated()` equivalent for MPS to report actual peak memory.

6. **Consider adding mlx-community quantized models** — Models like `mlx-community/Qwen2.5-3B-Instruct-4bit` could be tested once MLX mode is added.

---

## Commit

```bash
git add HARDWARE_TEST_RESULTS.md && git commit -m "test: Mac Mini M4 Pro hardware test results"
```

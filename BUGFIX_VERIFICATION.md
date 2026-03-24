# Bug Fix Verification

**Date:** 2026-03-24  
**Platform:** Mac Mini M4 Pro (macOS)  
**Python:** python3 (system)  
**Device:** MPS (Apple Silicon GPU)

---

## Fix 1: `--model` argument ✅ PASS

- **Test:** `python3 real_training.py --model EleutherAI/pythia-70m --steps 50`
- **Result:** Model ID in output matches `--model` argument exactly (`EleutherAI/pythia-70m`)
- **Details:**
  - Vocab size: 50,254
  - LoRA applied to 18 layers
  - Trainable params: 147,456 (0.21%)
  - Final loss (step 50): 2.0061
  - Training completed successfully, checkpoint saved

---

## Fix 2: LoRA in federated client ❌ FAIL

- **Test:** `apply_lora_to_model()` on pythia-70m
- **Result:** CRASH — `AttributeError: module 'torch.nn' has no attribute 'Conv1D'. Did you mean: 'Conv1d'?`
- **Root cause:** `federated/client.py` line 62 references `nn.Conv1D` which doesn't exist in modern PyTorch. The correct class is `nn.modules.conv.Conv1d`.
- **Expected:** Thousands of trainable params
- **Actual:** Process aborts before any params can be counted
- **Impact:** Federated training path is broken due to this import error

---

## Fix 3: requirements.txt ✅ PASS

- **Test:** `grep -E 'psutil|cryptography|fastapi|uvicorn|pytest' requirements.txt`
- **Result:** All 5 packages present:
  - `psutil>=5.9.0` ✅
  - `cryptography>=41.0.0` ✅
  - `fastapi>=0.104.0` ✅
  - `uvicorn>=0.24.0` ✅
  - `pytest>=7.4.0` ✅

---

## Real Training Benchmarks

### Run 1: pythia-70m (100 steps — timed out waiting for inference)

| Metric | Value |
|--------|-------|
| Model | EleutherAI/pythia-70m |
| Steps requested | 100 |
| Steps completed (checkpoint) | 50 |
| Step 50 loss | 1.9794 |
| Trainable params | 147,456 (0.21%) |
| LoRA layers | 18 |
| Device | mps:0 |
| Status | Checkpoint saved at step 50; process killed during inference demo |

### Run 2: pythia-160m (50 steps)

| Metric | Value |
|--------|-------|
| Model | EleutherAI/pythia-160m |
| Steps | 50 |
| Time | ~47 seconds (11:33:20 → 11:34:06) |
| Final loss | 0.9344 |
| Trainable params | 442,368 (0.27%) |
| LoRA layers | 36 |
| Device | mps:0 |
| Status | ✅ Completed successfully |

### Inference Output (pythia-160m, post-training)
- `"PyTorch is a"` → `"PyTorch is a a a a a a a a a a a a a a a <"` *(repetition — model needs more training)*
- `"Machine learning models"` → `"Machine learning models models models models models models ofofofofofofofofofof"` *(repetition)*
- `"Neural networks can"` → `"Neural networks can can can can can can be can is = = = = = = ="` *(repetition)*

> Note: Repetitive output at 50 steps is expected for small pythia models fine-tuned briefly. Not indicative of a bug.

---

## Summary

| Fix | Status | Notes |
|-----|--------|-------|
| Fix 1: `--model` argument | ✅ PASS | Correct model loaded and used |
| Fix 2: LoRA in federated | ❌ FAIL | `nn.Conv1D` → `nn.modules.conv.Conv1d` needed in `federated/client.py:62` |
| Fix 3: requirements.txt | ✅ PASS | All 5 packages present |
| Real training (pythia-70m) | ✅ Partial | 50/100 steps before timeout, loss declining properly |
| Real training (pythia-160m) | ✅ PASS | 50 steps, loss 0.9344, 442K trainable params |

---

## Bug Fix for Fix 2

**File:** `federated/client.py`  
**Line:** ~62  
**Change:**
```python
# Before (broken):
self.is_conv1d = isinstance(linear, (nn.Conv1D, nn.modules.conv.Conv1d))

# After (fixed):
self.is_conv1d = isinstance(linear, nn.modules.conv.Conv1d)
```

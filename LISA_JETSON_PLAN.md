# LISA-Jetson: Training Large Models on 7.4GB RAM

## Executive Summary

**Goal:** Train large language models (7B+) on Jetson Orin's 7.4GB RAM using memory-efficient techniques.

**Reality Check:** The original claim of "120B on 7.4GB RAM" requires clarification:
- **Training** 120B parameters = NOT POSSIBLE without multi-device setup
- **Inference** on 120B with layer streaming = POSSIBLE with MoE models
- **Fine-tuning** 7B with QLoRA = POSSIBLE with careful optimization

---

## Current System Analysis

### Hardware Capabilities
| Component | Specification |
|-----------|---------------|
| CPU | ARM Cortex-A78AE (8-core) |
| GPU | NVIDIA Ampere (integrated, 2048 cores) |
| RAM | 7.4 GB LPDDR5 |
| Storage | NVMe SSD (200GB+ free) |
| CUDA | 12.6 |

### Known Issues
1. **GPU memory allocation fails** - CUDA buffer errors
2. **PyTorch CUDA not properly configured** - integrated GPU sharing memory
3. **GGUF files corrupted** - 32B f16 showing as 7.3GB instead of 64GB

### What's Verified Working
| Approach | Model Size | Status |
|----------|------------|--------|
| Layer-by-layer + LoRA | 0.5B | ✅ WORKS (GPU: 0.06GB) |
| Ollama inference | 3B | ✅ WORKS |
| GGUF mmap loading | 14B | ⚠️ Loads but slow |

---

## Research Findings

### State-of-the-Art Techniques (2024)

1. **QLoRA (Quantized LoRA)**
   - 4-bit quantization + LoRA adapters
   - Train 65B model on 48GB GPU
   - For 7B: ~3.5GB quantized + 0.5GB LoRA = 4GB total

2. **FSDP (Fully Sharded Data Parallel)**
   - Shards model across multiple GPUs
   - Requires multiple GPUs (not applicable to single Jetson)

3. **CPU Offloading**
   - Keeps weights on CPU RAM
   - Copies layers to GPU during forward pass
   - Enables larger models but SLOW

4. **Gradient Checkpointing**
   - Recomputes activations instead of storing them
   - Reduces memory by ~60% but 30% slower

5. **llama.cpp + CPU Training**
   - llama.cpp supports CPU-based training mode
   - Can handle larger models via mmap
   - Very slow but works on CPU-only systems

6. **MoE (Mixture of Experts)**
   - Only activates subset of experts per token
   - Llama-3.3-70B: 397B total, ~10B active
   - Inference possible with expert streaming

---

## Implementation Plan

### Phase 1: Fix GPU Memory Allocation (CRITICAL)
**Goal:** Get PyTorch CUDA working properly

**Steps:**
1. [ ] Test CUDA availability and memory allocation
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.mem_get_info())
   torch.cuda.set_per_process_memory_fraction(0.5)
   ```

2. [ ] Try different CUDA memory allocators
   - Check if `PYTORCH_CUDA_ALLOC_CONF` helps
   - Try `torch.cuda.empty_cache()` before allocation

3. [ ] Alternative: Use CPU-only training with CUDA disabled

**Verification:** Run `python3 -c "import torch; t=torch.randn(100,100).cuda(); print(t.device)"`

---

### Phase 2: QLoRA 7B Training (TARGET)
**Goal:** Train Qwen2.5-7B with 4-bit quantization on Jetson

**Steps:**
1. [ ] Install required packages
   ```bash
   pip install bitsandbytes peft accelerate transformers
   ```

2. [ ] Create QLoRA training script with:
   - 4-bit NF4 quantization (bitsandbytes)
   - LoRA rank=4, alpha=8
   - Gradient checkpointing
   - CPU offload for optimizer states

3. [ ] Test with minimal config (seq_len=16, batch=1)

4. [ ] Measure memory usage at each stage

**Expected Memory Usage:**
| Component | Size |
|-----------|------|
| 7B @ 4-bit | 3.5 GB |
| LoRA params | 0.5 GB |
| Gradients (checkpointed) | 0.5 GB |
| Activations (recomputed) | 0.5 GB |
| Optimizer (CPU offload) | 0 GB |
| **Total** | **~5 GB** |

**Verification:** Training runs 10 steps without OOM

---

### Phase 3: llama.cpp CPU Training
**Goal:** Use llama.cpp for larger models via CPU training

**Steps:**
1. [ ] Check llama.cpp training support
   ```bash
   # llama.cpp has a training mode
   ./quantize --help | grep -i train
   ```

2. [ ] Prepare training data in correct format

3. [ ] Test with small model (3B) first

4. [ ] Run training with layer-by-layer processing

**Note:** llama.cpp training is newer and less documented

**Verification:** `./train-lora` runs on test data

---

### Phase 4: MoE 70B Inference (If Training Works)
**Goal:** Demonstrate 70B-class model capability

**Steps:**
1. [ ] Test llama.cpp with Llama-3.3-70B-Q4_K_M (40GB file)

2. [ ] Use mmap for lazy loading

3. [ ] Run inference benchmark

**Note:** This is INFERENCE only, not training

**Verification:** Model generates coherent output

---

### Phase 5: Full Layer-by-Layer Training (Advanced)
**Goal:** Implement true LISA layer-by-layer training

**Steps:**
1. [ ] Load model to RAM (not GPU)
2. [ ] Extract each transformer layer
3. [ ] Process one layer at a time:
   - Copy layer to GPU
   - Run forward pass
   - Copy layer back to CPU
   - Repeat for all layers
4. [ ] Aggregate gradients for LoRA updates

**Challenges:**
- Slow (each forward pass = 28 layer transfers for 7B)
- Need to handle embedding and output layers
- Gradient accumulation across layers

**Verification:** Loss decreases over 10 steps

---

## Testing Protocol

### Test Categories

1. **Memory Tests**
   - Measure RAM usage at each stage
   - Verify no GPU memory leaks
   - Check swap usage

2. **Functional Tests**
   - Forward pass completes
   - Backward pass completes
   - Optimizer step completes
   - Checkpoint saves correctly

3. **Quality Tests**
   - Loss decreases over time
   - Generated text is coherent
   - LoRA weights have non-zero gradients

### Test Schedule
| Phase | Test | Success Criteria |
|-------|------|------------------|
| 1 | CUDA allocation | No error |
| 2 | QLoRA 7B forward | No OOM |
| 2 | QLoRA 7B training | 10 steps complete |
| 3 | llama.cpp train | Loss decreases |
| 4 | 70B inference | Text generated |
| 5 | Full LISA | 10 layers processed |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| GPU memory still fails | HIGH | Use CPU-only mode |
| 7B still OOM | MEDIUM | Reduce to 3B or use extreme quantization |
| Training too slow | HIGH | Accept slow training as limitation |
| llama.cpp train fails | MEDIUM | Use PyTorch as fallback |

---

## Expected Outcomes

### Realistic Goals
1. ✅ **QLoRA 3B training** on Jetson (HIGH CONFIDENCE)
2. ⚠️ **QLoRA 7B training** on Jetson (MEDIUM CONFIDENCE)
3. ⚠️ **Layer-by-layer 7B** - Slow but possible (MEDIUM)
4. ✅ **70B MoE inference** via llama.cpp (HIGH)
5. ❌ **120B training** on 7.4GB - NOT POSSIBLE (realistic physics)

### What We CAN Claim
- "Training 3B models on Jetson with 7.4GB RAM"
- "Layer-by-layer training reduces GPU memory to 0.06GB"
- "70B MoE inference via expert streaming"
- "QLoRA enables fine-tuning on consumer hardware"

### What We CANNOT Claim (Without Multi-GPU)
- "Training 7B on 7.4GB" (without qualification)
- "Training 120B on Jetson" (unless using distributed setup)

---

## References

1. LISA Paper: https://arxiv.org/abs/2403.17919
2. QLoRA Paper: https://arxiv.org/abs/2305.14314
3. Answer.AI FSDP+QLoRA: https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
4. ENERZAi 8B on Jetson: https://medium.com/@enerzai/8b-llama-on-8gb-jetson
5. llama.cpp training: https://github.com/ggerganov/llama.cpp

---

*Plan Created: 2026-04-01*
*Last Updated: 2026-04-01*

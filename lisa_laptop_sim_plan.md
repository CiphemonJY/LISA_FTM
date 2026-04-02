# LISA Jetson Laptop Simulation Plan

## Goal
Train and infer LLMs on Jetson Orin (8GB RAM, GPU broken) as a proxy for basic gaming laptops.

**Reality Check:** CPU-only, 8GB RAM, no GPU = slow but doable with right techniques.

---

## Phase 1: Baseline Inference (Today)

### Target: Qwen2.5-1.5B on CPU

**Why:** Fits in 8GB RAM, fast enough for testing
**Memory:** ~3GB (4-bit) vs ~9GB (16-bit)

```bash
# Kill any running processes
ssh jetson@10.0.0.145 "pkill -f train; pkill -f ollama"

# Pull quantized model
ssh jetson@10.0.0.145 "cd ~/ollama && ./ollama pull qwen2.5:1.5b"

# Test inference
ssh jetson@10.0.0.145 "echo 'What is Python?' | ./ollama run qwen2.5:1.5b"
```

**Expected:** ~10-20 tokens/sec (slow but usable)

---

## Phase 2: LCSB Training Setup (Today)

### Install Requirements
```bash
ssh jetson@10.0.0.145
pip install transformers torch peft safetensors
```

### Create LCSB Training Script
```python
# lisa_lcsb_train.py - Layer-wise CPU-Sharded Backprop

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import gc

MODEL = "Qwen/Qwen2.5-1.5B"
RANK = 4
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

def train_lcsb(dataset_path):
    """Train with layer-wise loading to minimize RAM"""
    
    # Load tokenizer (small, ~10MB)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # Calculate memory per layer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, 
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    total_layers = len(model.model.layers)
    print(f"Total layers: {total_layers}")
    
    # Get layer memory footprint
    layer_memory = estimate_layer_memory(model) / 1024**2
    print(f"Each layer: ~{layer_memory:.1f} MB")
    
    # Setup LoRA (only these params train)
    lora_config = LoraConfig(r=RANK, target_modules=TARGET_MODULES)
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.2f}M)")
    
    # Training loop - load layers sequentially
    for epoch in range(10):
        for batch in dataloader:
            # Forward pass - layer by layer
            hidden_states = load_embeddings(batch)
            
            for layer_idx in range(total_layers):
                layer = model.model.layers[layer_idx]
                
                # Offload previous layers to disk
                if layer_idx > 0:
                    save_layer_to_disk(model.model.layers[layer_idx - 1])
                    del model.model.layers[layer_idx - 1]
                    gc.collect()
                
                # Load current layer to RAM
                layer = load_layer_from_disk(layer_idx)
                
                # Forward through this layer only
                hidden_states = layer(hidden_states)
                
            # Final layer norm + logits
            logits = model.lm_head(hidden_states)
            
            # Backward - only update LoRA on loaded layer
            loss.backward()
            
            # Save checkpoint every 50 rounds
            if round % 50 == 0:
                model.save_pretrained(f"checkpoint-{round}")
    
    model.save_pretrained("final-model")

# Memory estimation
def estimate_layer_memory(model):
    """Calculate RAM per layer"""
    layer = model.model.layers[0]
    return sum(p.numel() * p.element_size() for p in layer.parameters())

# Disk offload helpers
def save_layer_to_disk(layer):
    torch.save(layer.state_dict(), f"/tmp/layer_{layer.layer_idx}.pt")

def load_layer_from_disk(layer_idx):
    layer_state = torch.load(f"/tmp/layer_{layer_idx}.pt")
    layer = load_layer_module()
    layer.load_state_dict(layer_state)
    return layer
```

---

## Phase 3: Run Training (Tomorrow)

### Commands
```bash
# On Jetson
cd ~/lisa
python lisa_lcsb_train.py --data /tmp/code_strings.json --rounds 300

# Monitor
watch -n 5 'free -h && echo "Round: $(cat /tmp/train_round.txt)"'
```

### Expected Results
- **Trainable params:** ~500K (LoRA only)
- **Memory:** ~3-4GB (layer + LoRA)
- **Disk I/O:** Main bottleneck (~100x slower than GPU)
- **Time:** ~1 round/minute (5 hours for 300 rounds)

---

## Phase 4: Test Inference (After Training)

```bash
# Reload model with LoRA
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen2.5-1.5B')
model.load_adapter('final-model')
model.push_to_hub('your-username/lisa-1.5b-lora')
"

# Or run locally
./ollama run qwen2.5:1.5b "Explain async/await in Python:"
```

---

## Memory Budget (8GB Laptop Simulation)

| Component | RAM |
|-----------|-----|
| Model (4-bit, 1.5B) | 1.5GB |
| Current Layer | 0.5GB |
| LoRA Weights | 0.05GB |
| Activations | 0.5GB |
| Tokenizer + Overhead | 0.5GB |
| **Total** | **~3GB** |

**Free:** 5GB for OS + other processes

---

## Next Steps

1. **Now:** Test Qwen2.5-1.5B inference on Jetson
2. **Tonight:** Setup LCSB script, verify memory usage
3. **Tomorrow:** Run 300-round training
4. **After:** Compare to Qwen2.5-3B if 1.5B works

---

## If 1.5B Works, Scale to 3B

```python
# Same script, just change model
MODEL = "Qwen/Qwen2.5-3B"  # vs 1.5B
# 3B in 4-bit = ~2GB per layer

# Memory budget for 3B
# Current layer: ~1GB
# LoRA: 0.05GB
# Total: ~2-3GB
```

---

## Reality Check

**This is slow.** CPU-only training is ~100x slower than GPU. But it proves the concept works on consumer hardware.

**Target:** 
- 300 rounds in ~5 hours (1.5B)
- 300 rounds in ~10 hours (3B)

**Use case:** Prototype locally, train larger models in cloud.

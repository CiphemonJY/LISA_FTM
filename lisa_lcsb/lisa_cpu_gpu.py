#!/usr/bin/env python3
"""
7B LISA+LCSB - CPU Heavy, GPU Assist for Attention
Uses CPU for most computation, GPU only for attention layers
"""
import gc
import torch
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

LOG_FILE = "/tmp/lisa_cpu_gpu_log.txt"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("="*60)
log("7B LISA+LCSB - CPU + GPU Hybrid")
log("="*60)

# Load model - keep on CPU
log("\n[1] Loading model on CPU...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
log(f"Model loaded: {len(model.model.layers)} layers")

# LISA - freeze all, unfreeze last 2
log("\n[2] Applying LISA...")
for p in model.parameters():
    p.requires_grad = False
for p in model.model.layers[-2:].parameters():
    p.requires_grad = True

# LoRA
log("[3] Applying LoRA...")
model = get_peft_model(model, LoraConfig(
    r=1, lora_alpha=2,
    target_modules=["q_proj", "k_proj", "v_proj"]
))

# LISA on LoRA - only last 2 layers
log("[4] Setting up LISA on LoRA...")
for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if ("layers.26." in name or "layers.27." in name) and "lora_" in name:
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Trainable params: {trainable:,}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)

# Training data
TEXTS = [
    "The cat sat on the mat and purred softly",
    "Machine learning enables computers to understand",
    "The quick brown fox jumps over the lazy dog",
    "Artificial intelligence is transforming our world",
    "Neural networks learn from data patterns",
    "Deep learning powers modern AI applications",
]

LAYER_CYCLE = [27, 26]
TOTAL_STEPS = 100

log(f"\n[5] Training {TOTAL_STEPS} steps (CPU mode)...")
log("="*60)

step = 0
start_time = time.time()

try:
    while step < TOTAL_STEPS:
        step += 1
        text_idx = (step - 1) % len(TEXTS)
        layer = LAYER_CYCLE[(step - 1) % 2]
        
        t0 = time.time()
        inputs = tokenizer(TEXTS[text_idx], return_tensors="pt", max_length=8)
        
        out = model(**inputs, labels=inputs["input_ids"])
        t1 = time.time()
        
        optimizer.zero_grad()
        out.loss.backward()
        t2 = time.time()
        
        grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        t3 = time.time()
        
        loss = out.loss.item()
        
        log(f"Step {step}/{TOTAL_STEPS}: layer={layer}, loss={loss:.4f}, "
            f"fwd={t1-t0:.1f}s, bwd={t2-t1:.1f}s, opt={t3-t2:.3f}s")
        
        if step % 20 == 0:
            checkpoint = {
                "step": step,
                "loss": loss,
                "model_state": {k: v.cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
            }
            torch.save(checkpoint, f"/tmp/lisa_cpu_gpu_step_{step}.pt")
            log(f"  -> Checkpoint saved")
        
        gc.collect()

except Exception as e:
    log(f"Error: {e}")
    import traceback
    traceback.print_exc()

log("="*60)
log(f"DONE! Steps: {step}, Final loss: {loss}")
log(f"Total time: {time.time() - start_time:.1f}s")
log("="*60)

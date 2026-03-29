#!/usr/bin/env python3
"""
LISA + LoRA + LCSB with bf16 + smart offload
bfloat16 with careful memory management
"""
import gc
import torch
import time
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

LOG_FILE = "/tmp/lisa_bf16_log.txt"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("="*60)
log("LISA + LoRA + LCSB with bf16 + OFFLOAD")
log("="*60)

MODEL_NAME = "Qwen/Qwen2.5-14B"
log(f"Model: {MODEL_NAME}")

log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

offload_folder = "/tmp/model_bf16_offload"
os.makedirs(offload_folder, exist_ok=True)

log("\n[1] Loading bf16 model with sequential offload...")

# Sequential device map - one layer at a time on GPU, rest on CPU
device_map = {
    "embed_tokens": "cpu",
    "layers.0": "cpu", "layers.1": "cpu", "layers.2": "cpu", "layers.3": "cpu",
    "layers.4": "cpu", "layers.5": "cpu", "layers.6": "cpu", "layers.7": "cpu",
    "layers.8": "cpu", "layers.9": "cpu", "layers.10": "cpu", "layers.11": "cpu",
    "layers.12": "cpu", "layers.13": "cpu", "layers.14": "cpu", "layers.15": "cpu",
    "layers.16": "cpu", "layers.17": "cpu", "layers.18": "cpu", "layers.19": "cpu",
    "layers.20": "cpu", "layers.21": "cpu", "layers.22": "cpu", "layers.23": "cpu",
    "layers.24": "cpu", "layers.25": "cpu", "layers.26": "cpu", "layers.27": "cpu",
    "layers.28": "cpu", "layers.29": "cpu", "layers.30": "cpu", "layers.31": "cpu",
    "layers.32": "cpu", "layers.33": "cpu", "layers.34": "cpu", "layers.35": "cpu",
    "layers.36": "cpu", "layers.37": "cpu", "layers.38": "cpu", "layers.39": "cpu",
    "layers.40": "cpu", "layers.41": "cpu", "layers.42": "cpu", "layers.43": "cpu",
    "layers.44": "cpu", "layers.45": "cpu",
    "layers.46": "cuda",  # Train last 2 layers on GPU
    "layers.47": "cuda",
    "norm": "cuda",
    "lm_head": "cpu",
}

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        offload_folder=offload_folder,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
except Exception as e:
    log(f"Failed with sequential: {e}")
    log("Trying simple CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

log(f"Model loaded: {len(model.model.layers)} layers")

# Move to GPU
if torch.cuda.is_available():
    # Only move the GPU parts
    log("Setting up GPU training...")
else:
    log("Training on CPU only")

# LISA
log("\n[2] Applying LISA...")
num_layers = len(model.model.layers)
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

# LISA on LoRA
log("[4] Setting up LISA on LoRA...")
target_layers = [num_layers - 2, num_layers - 1]
for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if any(f"layers.{l}." in name for l in target_layers) and "lora_" in name:
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Trainable params: {trainable:,}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)

TEXTS = [
    "The cat sat on the mat",
    "Machine learning works",
    "AI is powerful",
]

LAYER_CYCLE = target_layers[::-1]
TOTAL_STEPS = 50

log(f"\n[5] Training {TOTAL_STEPS} steps...")
log("="*60)

step = 0
start_time = time.time()

try:
    while step < TOTAL_STEPS:
        step += 1
        text_idx = (step - 1) % len(TEXTS)
        layer = LAYER_CYCLE[(step - 1) % len(LAYER_CYCLE)]
        
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
            f"fwd={t1-t0:.1f}s, bwd={t2-t1:.1f}s")
        
        if step % 20 == 0:
            checkpoint = {
                "step": step,
                "loss": loss,
                "model_state": {k: v.clone() for k, v in model.named_parameters() if v.requires_grad}
            }
            torch.save(checkpoint, f"/tmp/lisa_bf16_step_{step}.pt")
            log(f"  -> Checkpoint saved")
        
        gc.collect()

except Exception as e:
    log(f"Error at step {step}: {e}")
    import traceback
    traceback.print_exc()

log("="*60)
log(f"DONE! Steps: {step}, Final loss: {loss}")
log(f"Total time: {time.time() - start_time:.1f}s")
log("="*60)

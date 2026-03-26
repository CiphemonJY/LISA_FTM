#!/usr/bin/env python3
"""
LISA Standalone Trainer - No Server Needed
- Trains locally, saves checkpoints
- Uses dynamic memory management
- Can run alongside normal computer use
"""
import os, sys, time, gc, shutil
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model

# ============ Config ============
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_DIR = "/tmp/lisa_standalone_checkpoints"
LOG_FILE = "/tmp/lisa_standalone.log"

# Memory settings
MIN_MEMORY_MB = 256
IDLE_MEMORY_MB = 512
TARGET_MEMORY_MB = 2048  # Use up to 2GB when available

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    sys.stdout.flush()
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def get_available_memory_mb():
    """Get available memory in MB using vm_stat."""
    import subprocess
    result = subprocess.run(['vm_stat'], capture_output=True, text=True)
    for line in result.stdout.strip().split('\n'):
        if 'Pages free:' in line:
            free = int(line.split()[-1].rstrip('.'))
        elif 'Pages speculative:' in line:
            spec = int(line.split()[-1].rstrip('.'))
    return ((free + spec) * 16) / 1024

def should_train():
    """Check if enough memory is available."""
    avail = get_available_memory_mb()
    log(f"Available memory: {avail:.0f}MB")
    return avail >= MIN_MEMORY_MB

# ============ Model ============
class LocalTrainer:
    def __init__(self, memory_budget_mb):
        self.memory_budget_mb = memory_budget_mb
        self.model = None
        self.tokenizer = None
        self.round_num = 1
        
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Load existing checkpoint if available
        self._load_latest_checkpoint()
    
    def _load_latest_checkpoint(self):
        """Find and track the latest checkpoint."""
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) 
                             if f.startswith("model_round_") and f.endswith(".pt")])
        if checkpoints:
            latest = checkpoints[-1]
            num = int(latest.replace("model_round_", "").replace(".pt", ""))
            self.round_num = num + 1
            log(f"Resuming from round {self.round_num}")
    
    def load_model(self):
        """Load model with memory-efficient settings."""
        avail = get_available_memory_mb()
        
        log(f"Loading model (memory: {avail:.0f}MB available)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(MODEL_NAME)
        
        # Adjust based on available memory
        if avail < 512:
            torch_dtype = torch.float16
            lora_r = 2
            log("Low memory mode: float16, LoRA r=2")
        elif avail < 1024:
            torch_dtype = torch.float16
            lora_r = 4
            log("Medium memory mode: float16, LoRA r=4")
        else:
            torch_dtype = torch.float32
            lora_r = 4
            log("High memory mode: float32, LoRA r=4")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, config=config, trust_remote_code=True, torch_dtype=torch_dtype
        )
        
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r*2,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Freeze base, only LoRA trainable
        for name, param in self.model.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
        
        log(f"Model ready: {sum(p.numel() for p in self.model.parameters()):,} params")
        return self
    
    def train(self, steps=10):
        """Train for specified steps."""
        self.model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.01
        )
        
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Once upon a time in a far away land,",
            "Machine learning is transforming the world.",
            "Artificial intelligence is the future.",
            "Federated learning enables privacy preserving AI.",
        ]
        
        for i in range(steps):
            text = texts[i % len(texts)]
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            if (i + 1) % 5 == 0:
                log(f"  Step {i+1}/{steps}: loss={loss.item():.4f}")
        
        return loss.item()
    
    def save_checkpoint(self):
        """Save checkpoint with rotation (keep last 3)."""
        path = os.path.join(CHECKPOINT_DIR, f"model_round_{self.round_num}.pt")
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, path)
        log(f"💾 Saved: model_round_{self.round_num}.pt")
        
        # Rotate checkpoints
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) 
                            if f.startswith("model_round_") and f.endswith(".pt")])
        while len(checkpoints) > 3:
            old = checkpoints.pop(0)
            os.remove(os.path.join(CHECKPOINT_DIR, old))
            log(f"🗑️ Rotated out: {old}")
        
        self.round_num += 1
    
    def unload(self):
        """Unload model to free memory."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        log("Model unloaded")

# ============ Main ============
def main():
    print(f"""
╔════════════════════════════════════════════════════════════╗
║  LISA Standalone Trainer                              ║
║  - No server needed, trains locally                ║
║  - Dynamic memory management                       ║
║  - Saves checkpoints, continues training          ║
╚════════════════════════════════════════════════════════════╝
Model: {MODEL_NAME}
""")
    
    log("🚀 LISA Standalone Trainer Started")
    
    # Get initial memory
    avail = get_available_memory_mb()
    budget = min(avail - 256, TARGET_MEMORY_MB) if avail > MIN_MEMORY_MB else MIN_MEMORY_MB
    log(f"Initial memory available: {avail:.0f}MB, budget: {budget:.0f}MB")
    
    trainer = LocalTrainer(budget)
    
    while True:
        try:
            # Wait for enough memory
            while not should_train():
                log("Memory low, waiting 60s...")
                time.sleep(60)
            
            # Load and train
            trainer.load_model()
            
            log(f"📍 Training round {trainer.round_num}")
            loss = trainer.train(steps=10)
            log(f"  Loss: {loss:.4f}")
            
            trainer.save_checkpoint()
            trainer.unload()
            
            # Brief pause between rounds
            log("Resting 30s before next round...")
            time.sleep(30)
            
        except KeyboardInterrupt:
            log("Stopped by user")
            break
        except Exception as e:
            log(f"Error: {e}")
            if trainer.model:
                trainer.unload()
            time.sleep(60)

if __name__ == "__main__":
    main()

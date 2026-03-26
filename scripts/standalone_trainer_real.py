#!/usr/bin/env python3
"""
LISA Standalone Trainer with Real Dataset
- Uses wikitext dataset for meaningful training
- Dynamic memory management
- Checkpoint rotation
"""
import os, sys, time, gc
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# ============ Config ============
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_NAME = "wikitext"  # Real text dataset
CHECKPOINT_DIR = "/tmp/lisa_standalone_checkpoints"
LOG_FILE = "/tmp/lisa_standalone.log"
MAX_STEPS = 20  # More steps per round
LOG_INTERVAL = 5

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()
    with open(LOG_FILE, "a") as f:
        f.write(f"[{ts}] {msg}\n")

def get_available_memory_mb():
    import subprocess
    result = subprocess.run(['vm_stat'], capture_output=True, text=True)
    for line in result.stdout.strip().split('\n'):
        if 'Pages free:' in line:
            free = int(line.split()[-1].rstrip('.'))
        elif 'Pages speculative:' in line:
            spec = int(line.split()[-1].rstrip('.'))
    return ((free + spec) * 16) / 1024

def load_dataset_texts():
    """Load real text dataset."""
    log(f"Loading dataset: {DATASET_NAME}...")
    try:
        dataset = load_dataset(DATASET_NAME, "wikitext-2-v1", split="train")
        texts = [item['text'] for item in dataset if item['text'].strip() and len(item['text'].strip()) > 20]
        log(f"Loaded {len(texts)} text samples")
        return texts
    except Exception as e:
        log(f"Dataset error: {e}")
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Once upon a time in a far away land there lived a brave hero.",
            "Machine learning is transforming the world in profound ways.",
            "Artificial intelligence enables computers to learn from experience.",
        ]

class Trainer:
    def __init__(self):
        self.round_num = 1
        self.texts = None  # Lazy load
        self.model = None
        self.tokenizer = None
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Find latest checkpoint
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
        if checkpoints:
            self.round_num = int(checkpoints[-1].replace('model_round_','').replace('.pt','')) + 1
            log(f"Resuming from round {self.round_num}")
    
    def load_data(self):
        """Load dataset once."""
        if self.texts is None:
            self.texts = load_dataset_texts()
    
    def load_model(self):
        avail = get_available_memory_mb()
        log(f"Loading model ({avail:.0f}MB available)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(MODEL_NAME)
        
        if avail < 1024:
            torch_dtype = torch.float16
            lora_r = 2
        else:
            torch_dtype = torch.float32
            lora_r = 4
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, config=config, trust_remote_code=True, torch_dtype=torch_dtype
        )
        
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r*2,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        
        for n, p in self.model.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False
        
        log(f"Model ready: {sum(p.numel() for p in self.model.parameters()):,} params")
        return self
    
    def train(self, steps=MAX_STEPS):
        self.load_data()
        self.model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001
        )
        
        total_loss = 0
        for i in range(steps):
            # Get random text from dataset
            import random
            text = random.choice(self.texts)
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
            if (i + 1) % LOG_INTERVAL == 0:
                log(f"  Step {i+1}/{steps}: loss={loss.item():.4f}")
        
        avg_loss = total_loss / steps
        return avg_loss
    
    def save(self):
        path = f"{CHECKPOINT_DIR}/model_round_{self.round_num}.pt"
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, path)
        log(f"💾 Saved round_{self.round_num} (loss={self.avg_loss:.4f})")
        
        # Rotate - keep 3
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
        while len(checkpoints) > 3:
            os.remove(f"{CHECKPOINT_DIR}/{checkpoints.pop(0)}")
        
        self.round_num += 1
    
    def unload(self):
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        gc.collect()
        log("Model unloaded")

def main():
    log(f"""
╔════════════════════════════════════════════════════════════╗
║  LISA Standalone + Real Dataset                      ║
╚════════════════════════════════════════════════════════════╝
Model: {MODEL_NAME}
Dataset: {DATASET_NAME}
Steps per round: {MAX_STEPS}
""")
    
    trainer = Trainer()
    
    while True:
        try:
            avail = get_available_memory_mb()
            if avail < 256:
                log(f"Memory low ({avail:.0f}MB), waiting...")
                time.sleep(60)
                continue
            
            trainer.load_model()
            log(f"📍 Round {trainer.round_num}")
            
            loss = trainer.train()
            trainer.avg_loss = loss
            log(f"  Avg loss: {loss:.4f}")
            
            trainer.save()
            trainer.unload()
            
            log(f"Resting 60s...")
            time.sleep(60)
            
        except KeyboardInterrupt:
            log("Stopped")
            break
        except Exception as e:
            log(f"Error: {e}")
            if trainer.model:
                trainer.unload()
            time.sleep(60)

if __name__ == "__main__":
    main()

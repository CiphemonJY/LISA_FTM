#!/usr/bin/env python3
"""LISA Jetson Standalone Trainer"""
import os, sys, time, gc
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_DIR = "/tmp/lisa_jetson_checkpoints"
LOG_FILE = "/tmp/lisa_jetson_standalone.log"

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

def get_gpu_memory_mb():
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip())
    except:
        return 4096

class Trainer:
    def __init__(self):
        self.round_num = 1
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        # Load latest checkpoint if exists
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
        if checkpoints:
            self.round_num = int(checkpoints[-1].replace('model_round_','').replace('.pt','')) + 1
            log(f"Resuming from round {self.round_num}")
    
    def load_model(self):
        gpu_mem = get_gpu_memory_mb()
        log(f"Loading model ({gpu_mem}MB GPU free)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(MODEL_NAME)
        
        if gpu_mem < 2048:
            torch_dtype = torch.float16
            lora_r = 2
        else:
            torch_dtype = torch.float32
            lora_r = 4
        
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, trust_remote_code=True, torch_dtype=torch_dtype)
        
        lora_config = LoraConfig(r=lora_r, lora_alpha=lora_r*2, target_modules=["q_proj", "k_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(self.model, lora_config)
        
        for n, p in self.model.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False
        
        log(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
        return self
    
    def train(self, steps=10):
        self.model.train()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.01)
        
        texts = ["The quick brown fox jumps over the lazy dog.", "Once upon a time in a far away land.", "Machine learning is transforming the world.", "AI is the future."]
        
        for i in range(steps):
            inputs = self.tokenizer(texts[i%4], return_tensors="pt", truncation=True, max_length=64)
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            if (i+1) % 5 == 0:
                log(f"  Step {i+1}: loss={loss.item():.4f}")
        
        return loss.item()
    
    def save(self):
        path = f"{CHECKPOINT_DIR}/model_round_{self.round_num}.pt"
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, path)
        log(f"💾 Saved round_{self.round_num}")
        
        # Rotate - keep 3
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
        while len(checkpoints) > 3:
            os.remove(f"{CHECKPOINT_DIR}/{checkpoints.pop(0)}")
        
        self.round_num += 1
    
    def unload(self):
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        gc.collect()

def main():
    log("🚀 Jetson Standalone Trainer Started")
    trainer = Trainer()
    
    while True:
        try:
            trainer.load_model()
            loss = trainer.train()
            log(f"Round {trainer.round_num}: loss={loss:.4f}")
            trainer.save()
            trainer.unload()
            log("Resting 60s...")
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

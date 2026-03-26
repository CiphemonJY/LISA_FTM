#!/usr/bin/env python3
import os, sys, time, gc, random
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

MODEL = "Qwen/Qwen2.5-0.5B"
DIR = "/tmp/lisa_jetson_checkpoints"
LOG = "/tmp/lisa_jetson_real.log"
STEPS = 20

def log(m):
    t = datetime.now().strftime("%H:%M:%S")
    x = "[" + t + "] " + m + chr(10)
    print(x, end="")
    sys.stdout.flush()
    open(LOG, "a").write(x)

def gpu_mem():
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    val = r.stdout.strip().split()[0] if r.stdout.strip() else "4096"
    try:
        return int(val.replace("[","").replace("]",""))
    except:
        return 4096

def load_ds():
    log("Loading wikitext...")
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    texts = [item["text"] for item in ds if item["text"].strip() and len(item["text"].strip()) > 20]
    log("Loaded " + str(len(texts)))
    return texts

class T:
    def __init__(self):
        self.r = 1
        self.txt = load_ds()
        os.makedirs(DIR, exist_ok=True)
        cps = sorted([f for f in os.listdir(DIR) if f.endswith(".pt")])
        if cps:
            self.r = int(cps[-1].replace("model_round_","").replace(".pt","")) + 1
            log("Resume r" + str(self.r))
    
    def load(self):
        g = gpu_mem()
        log("Load " + str(g) + "MB")
        self.tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
        self.tok.pad_token = self.tok.eos_token
        dtype = torch.float16 if g < 2048 else torch.float32
        log("dtype=" + str(dtype))
        self.m = AutoModelForCausalLM.from_pretrained(MODEL, config=AutoConfig.from_pretrained(MODEL), trust_remote_code=True, torch_dtype=dtype)
        self.m = get_peft_model(self.m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj","k_proj","v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
        for n, p in self.m.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False
        log("Ready")
        return self
    
    def train(self):
        self.m.train()
        o = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.m.parameters()), lr=0.001)
        tot = 0
        for i in range(STEPS):
            txt = random.choice(self.txt)
            inp = self.tok(txt, return_tensors="pt", truncation=True, max_length=128)
            o.zero_grad()
            loss = self.m(**inp, labels=inp["input_ids"]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
            o.step()
            tot += loss.item()
            if (i+1) % 5 == 0:
                log("S" + str(i+1) + " loss=" + str(round(loss.item(), 4)))
        return tot / STEPS
    
    def save(self):
        torch.save({k: v.cpu() for k, v in self.m.state_dict().items()}, DIR + "/model_round_" + str(self.r) + ".pt")
        log("Saved r" + str(self.r))
        cps = sorted([f for f in os.listdir(DIR) if f.endswith(".pt")])
        while len(cps) > 3:
            os.remove(DIR + "/" + cps.pop(0))
        self.r += 1
    
    def free(self):
        del self.m
        gc.collect()

def main():
    log("Jetson wikitext start")
    t = T()
    while True:
        try:
            t.load()
            loss = t.train()
            log("R" + str(t.r) + " loss=" + str(round(loss, 4)))
            t.save()
            t.free()
            time.sleep(60)
        except Exception as e:
            log("Err: " + str(e))
            time.sleep(60)

main()

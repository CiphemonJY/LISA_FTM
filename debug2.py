#!/usr/bin/env python3
"""
Minimal 1-client test: directly train for 20 steps vs federated snapshot-then-apply-delta.
Compare ppl change. This isolates the federation bug.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---- LoRA ----
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=8.0, dropout=0.05):
        super().__init__()
        self.weight_data = linear.weight.data.clone().float()
        self.bias_data = linear.bias.data.clone().float() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight_data.shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        with torch.no_grad():
            original = nn.functional.linear(x_f32, self.weight_data, self.bias_data)
        lora = nn.functional.linear(self.lora_dropout(x_f32), self.lora_A)
        lora = nn.functional.linear(lora, self.lora_B)
        return (original + lora * self.scaling).to(orig_dtype)


class LoraModel:
    TARGETS = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
               "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc"]

    def __init__(self, model, rank=4, alpha=8.0):
        self.model = model
        self.layers = {}
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not any(t in name.split('.')[-1] for t in self.TARGETS):
                continue
            lora = LoRALinear(mod, rank=rank, alpha=alpha)
            parent_name, attr = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), attr, lora)
            self.layers[name] = lora

    def trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for l in self.layers.values():
            for p in l.parameters():
                p.requires_grad = True

    def snapshot(self):
        return {f"{k}.A": l.lora_A.data.clone() for k, l in self.layers.items()} | \
               {f"{k}.B": l.lora_B.data.clone() for k, l in self.layers.items()}

    def restore(self, snap):
        for k, l in self.layers.items():
            l.lora_A.data.copy_(snap[f"{k}.A"].clone())
            l.lora_B.data.copy_(snap[f"{k}.B"].clone())

    def apply_delta(self, delta, alpha=1.0):
        for k, l in self.layers.items():
            l.lora_A.data.add_(delta[f"{k}.A"], alpha=alpha)
            l.lora_B.data.add_(delta[f"{k}.B"], alpha=alpha)


# ---- Helpers ----
MODEL_ID = "EleutherAI/pythia-70m"
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
NUM_TRAIN_BATCHES = 20
MAX_TEST_BATCHES = 10

def tokenize(tokenizer, texts):
    enc = tokenizer(texts, max_length=MAX_SEQ_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc

def ppl(model, test_enc, tokenizer):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pad_id = getattr(tokenizer, 'pad_token_id', None) or -100
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    for i in range(min(MAX_TEST_BATCHES, len(test_enc["input_ids"]) // BATCH_SIZE)):
        s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        ids = test_enc["input_ids"][s:e].clone()
        labs = test_enc["labels"][s:e].clone()
        with torch.no_grad():
            out = model(input_ids=ids)
        loss = criterion(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


# ---- Main ----
print("Loading...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg,
             trust_remote_code=True, torch_dtype=torch.float32)
lm = LoraModel(model, rank=4, alpha=8.0)
print(f"  LoRA layers: {len(lm.layers)}")

print("Loading wikitext...")
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    ts = load_dataset("wikitext", "wikitext-2-v1", split="test")
    train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
    test_texts = [t for t in ts["text"] if t.strip() and len(t.strip()) > 20]
    print(f"  {len(train_texts)} train, {len(test_texts)} test")
except:
    train_texts = [" ".join(["word"]*30) for _ in range(200)]
    test_texts = [" ".join(["word"]*30) for _ in range(50)]

train_enc = tokenize(tok, train_texts)
test_enc = tokenize(tok, test_texts)

LR = 3e-4

# Test A: Direct training, 20 steps, measure ppl before and after
print("\n--- Test A: Direct training (20 steps) ---")
lm.freeze_all(); lm.unfreeze_all()
opt = torch.optim.AdamW(lm.trainable_params(), lr=LR, weight_decay=0.01)
ppl_before_A = ppl(model, test_enc, tok)
print(f"  ppl before: {ppl_before_A:.2f}")

for i in range(NUM_TRAIN_BATCHES):
    idx = torch.randperm(len(train_enc["input_ids"]))[:BATCH_SIZE].tolist()
    ids = train_enc["input_ids"][idx].clone().clamp(0, tok.vocab_size - 1)
    labs = train_enc["labels"][idx].clone().clamp(0, tok.vocab_size - 1)
    opt.zero_grad()
    out = model(input_ids=ids)
    loss = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id or -100)(
        out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lm.trainable_params(), 1.0)
    opt.step()

ppl_after_A = ppl(model, test_enc, tok)
print(f"  ppl after:  {ppl_after_A:.2f}")
print(f"  delta:      {ppl_after_A - ppl_before_A:+.2f}")

# Test B: Federated style — snapshot, train, delta, apply
print("\n--- Test B: Federated (snapshot -> train -> apply delta) ---")
# Reset
for l in lm.layers.values():
    nn.init.normal_(l.lora_A.data, std=0.01)
    nn.init.zeros_(l.lora_B.data)
lm.freeze_all(); lm.unfreeze_all()
opt = torch.optim.AdamW(lm.trainable_params(), lr=LR, weight_decay=0.01)
ppl_before_B = ppl(model, test_enc, tok)
print(f"  ppl before: {ppl_before_B:.2f}")

snap = lm.snapshot()
for i in range(NUM_TRAIN_BATCHES):
    idx = torch.randperm(len(train_enc["input_ids"]))[:BATCH_SIZE].tolist()
    ids = train_enc["input_ids"][idx].clone().clamp(0, tok.vocab_size - 1)
    labs = train_enc["labels"][idx].clone().clamp(0, tok.vocab_size - 1)
    opt.zero_grad()
    out = model(input_ids=ids)
    loss = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id or -100)(
        out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lm.trainable_params(), 1.0)
    opt.step()

snap_after = lm.snapshot()
delta = {k: snap_after[k] - snap[k] for k in snap}

# Reset to original snapshot (server starts from same state as before training)
lm.restore(snap)

# Apply delta with different server_lr values
for srv_lr in [1.0, 0.1]:
    lm.restore(snap)  # reset to original
    lm.apply_delta(delta, alpha=srv_lr)
    ppl_after = ppl(model, test_enc, tok)
    print(f"  SERVER_LR={srv_lr}: ppl after apply={ppl_after:.2f}  delta={ppl_after-ppl_before_B:+.2f}")

print("\n--- Result ---")
print(f"Direct training delta:       {ppl_after_A - ppl_before_A:+.2f}")
print("If SERVER_LR=1.0 federated delta differs -> federation is broken")
print("If SERVER_LR=0.1 matches better -> 0.1 is the right scaling")

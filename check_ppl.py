#!/usr/bin/env python3
"""Check ppl computation discrepancy between debug6 and quick_lr_test."""
import math, torch, torch.nn as nn, random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

MODEL_ID = "EleutherAI/pythia-70m"
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)

print(f"model.config.pad_token_id: {getattr(model.config, 'pad_token_id', 'NOT SET')}")
print(f"tokenizer.pad_token_id: {tok.pad_token_id}")

# Load wikitext test
ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
test_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
enc = tok(test_texts[:100], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
enc["labels"] = enc["input_ids"].clone()

BATCH_SIZE = 4
MAX_TEST = 10

# Method 1: debug6's approach (ignore_index = model.config.pad_token_id or 0)
model.eval()
total_loss1, total_tokens1 = 0.0, 0
pad_id1 = getattr(model.config, "pad_token_id", None) or 0
print(f"\nMethod 1 (debug6): ignore_index={pad_id1}")
crit1 = nn.CrossEntropyLoss(ignore_index=pad_id1)
for i in range(min(MAX_TEST, len(enc["input_ids"]) // BATCH_SIZE)):
    s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
    ids = enc["input_ids"][s:e].clone()
    labs = enc["labels"][s:e].clone()
    with torch.no_grad():
        out = model(input_ids=ids)
    loss = crit1(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
    total_loss1 += loss.item() * ids.numel()
    total_tokens1 += ids.numel()
ppl1 = math.exp(total_loss1 / max(total_tokens1, 1))
print(f"  total_loss={total_loss1:.4f}, total_tokens={total_tokens1}")
print(f"  ppl = {ppl1}")

# Method 2: quick_lr_test's approach (ignore_index = tok.pad_token_id or 0)
total_loss2, total_tokens2 = 0.0, 0
pad_id2 = tok.pad_token_id or 0
print(f"\nMethod 2 (quick_lr_test): ignore_index={pad_id2}")
crit2 = nn.CrossEntropyLoss(ignore_index=pad_id2)
for i in range(min(MAX_TEST, len(enc["input_ids"]) // BATCH_SIZE)):
    s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
    ids = enc["input_ids"][s:e].clone()
    labs = enc["labels"][s:e].clone()
    with torch.no_grad():
        out = model(input_ids=ids)
    loss = crit2(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
    total_loss2 += loss.item() * ids.numel()
    total_tokens2 += ids.numel()
ppl2 = math.exp(total_loss2 / max(total_tokens2, 1))
print(f"  total_loss={total_loss2:.4f}, total_tokens={total_tokens2}")
print(f"  ppl = {ppl2}")

# Method 3: no ignore_index (standard)
total_loss3, total_tokens3 = 0.0, 0
print(f"\nMethod 3 (no ignore_index):")
for i in range(min(MAX_TEST, len(enc["input_ids"]) // BATCH_SIZE)):
    s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
    ids = enc["input_ids"][s:e].clone()
    labs = enc["labels"][s:e].clone()
    with torch.no_grad():
        out = model(input_ids=ids)
    loss = nn.functional.cross_entropy(
        out.logits.view(-1, out.logits.size(-1)), labs.view(-1), reduction="sum")
    total_loss3 += loss.item()
    total_tokens3 += labs.numel()
ppl3 = math.exp(total_loss3 / max(total_tokens3, 1))
print(f"  total_loss={total_loss3:.4f}, total_tokens={total_tokens3}")
print(f"  ppl = {ppl3}")

print(f"\nConclusion:")
print(f"  Method 1 (pad_id=None->0): ppl={ppl1:.2f}")
print(f"  Method 2 (pad_id=1): ppl={ppl2:.2f}")
print(f"  Method 3 (no ignore): ppl={ppl3:.2f}")

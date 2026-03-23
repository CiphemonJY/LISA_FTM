#!/usr/bin/env python3
"""
Debug why FedAvg diverges. Train 1 client for 20 steps, apply delta, check perplexity.

Key question: is the bug in (a) how we compute deltas, (b) how we apply them,
or (c) something else entirely?
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---------------------------------------------------------------------------
# Inline minimal LoRA (same as byzantine_stress_test.py)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Module, rank=4, alpha=8.0, dropout=0.05):
        super().__init__()
        self.weight_data = linear.weight.data.clone().float()
        self.bias_data = linear.bias.data.clone().float() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight_data.shape
        self.is_conv1d = isinstance(linear, nn.Conv1d)
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
        lora_input = self.lora_dropout(x_f32)
        lora = nn.functional.linear(lora_input, self.lora_A)
        lora = nn.functional.linear(lora, self.lora_B)
        return (original + lora * self.scaling).to(orig_dtype)


class LoraAppliedModel:
    TARGET_MODULES = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
                      "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc"]

    def __init__(self, model, rank=4, alpha=8.0, dropout=0.05):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.lora_layers = {}

    def apply_lora(self):
        count = 0
        for full_name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d)):
                continue
            name_parts = full_name.split(".")
            if not any(tm in name_parts[-1] for tm in self.TARGET_MODULES):
                continue
            lora = LoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
            self.lora_layers[full_name] = lora
            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                try:
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, attr, lora)
                    count += 1
                except KeyError:
                    pass
        print(f"  LoRA applied to {count} layers")
        return count

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all_lora(self):
        for lora_layer in self.lora_layers.values():
            for p in lora_layer.parameters():
                p.requires_grad = True

    def snapshot(self):
        state = {}
        for full_name, lora_layer in self.lora_layers.items():
            state[f"{full_name}.lora_A"] = lora_layer.lora_A.data.clone().cpu()
            state[f"{full_name}.lora_B"] = lora_layer.lora_B.data.clone().cpu()
        return state

    def restore(self, state):
        for full_name, lora_layer in self.lora_layers.items():
            lora_layer.lora_A.data.copy_(state[f"{full_name}.lora_A"].clone())
            lora_layer.lora_B.data.copy_(state[f"{full_name}.lora_B"].clone())

    def get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Model + data
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/pythia-70m"
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
LR = 3e-4
TRAIN_BATCHES = 20
MAX_TEST_BATCHES = 10

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)

print("Applying LoRA...")
wrapper = LoraAppliedModel(model, rank=4, alpha=8.0)
wrapper.apply_lora()
wrapper.freeze_all()
wrapper.unfreeze_all_lora()

# Load wikitext (synthetic fallback)
print("Loading data...")
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    test_ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
    test_texts = [t for t in test_ds["text"] if t.strip() and len(t.strip()) > 20]
    print(f"  Real wikitext: {len(train_texts)} train, {len(test_texts)} test")
except Exception as e:
    print(f"  Using synthetic: {e}")
    train_texts = [" ".join(["word"] * 30) for _ in range(200)]
    test_texts = [" ".join(["word"] * 30) for _ in range(50)]

# Tokenize
def tokenize(texts):
    enc = tok(texts, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": enc["input_ids"].clone(),
    }

train_enc = tokenize(train_texts)
test_enc = tokenize(test_texts)

# Perplexity
def ppl(model):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pad = tok.pad_token_id if tok.pad_token_id is not None else -100
    for i in range(min(MAX_TEST_BATCHES, len(test_enc["input_ids"]) // BATCH_SIZE)):
        s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        ids = test_enc["input_ids"][s:e].clone()
        mask = test_enc["attention_mask"][s:e]
        labs = test_enc["labels"][s:e].clone()
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad)
        loss = loss_fn(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))

# Initial perplexity
print(f"\nInitial perplexity: {ppl(model):.2f}")

# ---- TEST 1: Single-client direct training (no federation) ----
print("\n" + "="*60)
print("TEST 1: Direct training (1 client, 20 steps)")
print("="*60)

# Reset to fresh model
for lora_layer in wrapper.lora_layers.values():
    nn.init.normal_(lora_layer.lora_A.data, mean=0, std=0.01)
    nn.init.zeros_(lora_layer.lora_B.data)

optimizer = torch.optim.AdamW(wrapper.get_trainable_params(), lr=LR, weight_decay=0.01)
for step in range(TRAIN_BATCHES):
    idx = torch.randperm(len(train_enc["input_ids"]))[:BATCH_SIZE].tolist()
    ids = train_enc["input_ids"][idx].clone()
    labs = train_enc["labels"][idx].clone()
    ids.clamp_(0, tok.vocab_size - 1)
    labs.clamp_(0, tok.vocab_size - 1)
    optimizer.zero_grad()
    out = model(input_ids=ids)
    loss = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id or -100)(
        out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wrapper.get_trainable_params(), 1.0)
    optimizer.step()
    if step % 5 == 0:
        print(f"  step {step}: loss={loss.item():.4f}")

print(f"  After training ppl: {ppl(model):.2f}")

# ---- TEST 2: Federated style (snapshot -> train -> delta -> apply) ----
print("\n" + "="*60)
print("TEST 2: Federated style (snapshot -> train -> apply delta)")
print("="*60)

# Reset model
for lora_layer in wrapper.lora_layers.values():
    nn.init.normal_(lora_layer.lora_A.data, mean=0, std=0.01)
    nn.init.zeros_(lora_layer.lora_B.data)
optimizer = torch.optim.AdamW(wrapper.get_trainable_params(), lr=LR, weight_decay=0.01)

before = wrapper.snapshot()
print(f"  Snapshot lora_A norm: {before[list(before.keys())[0]].norm():.4f}")

for step in range(TRAIN_BATCHES):
    idx = torch.randperm(len(train_enc["input_ids"]))[:BATCH_SIZE].tolist()
    ids = train_enc["input_ids"][idx].clone()
    labs = train_enc["labels"][idx].clone()
    ids.clamp_(0, tok.vocab_size - 1)
    labs.clamp_(0, tok.vocab_size - 1)
    optimizer.zero_grad()
    out = model(input_ids=ids)
    loss = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id or -100)(
        out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wrapper.get_trainable_params(), 1.0)
    optimizer.step()
    if step % 5 == 0:
        print(f"  step {step}: loss={loss.item():.4f}")

after = wrapper.snapshot()
delta = {k: after[k] - before[k] for k in before}

# Show delta magnitudes
delta_norms = {k: v.norm().item() for k, v in delta.items()}
avg_delta_norm = sum(delta_norms.values()) / len(delta_norms)
print(f"  Avg delta norm: {avg_delta_norm:.6f}")
print(f"  Max delta norm: {max(delta_norms.values()):.6f}")

# Reset model again and apply delta with different SERVER_LR values
for srv_lr in [1.0, 0.1, 0.01, 0.001]:
    for lora_layer in wrapper.lora_layers.values():
        nn.init.normal_(lora_layer.lora_A.data, mean=0, std=0.01)
        nn.init.zeros_(lora_layer.lora_B.data)

    # Apply delta with this SERVER_LR
    for full_name, lora_layer in wrapper.lora_layers.items():
        lora_layer.lora_A.data.add_(delta[f"{full_name}.lora_A"], alpha=srv_lr)
        lora_layer.lora_B.data.add_(delta[f"{full_name}.lora_B"], alpha=srv_lr)

    ppl_after = ppl(model)
    print(f"  SERVER_LR={srv_lr}: ppl={ppl_after:.2f} (should be ~{ppl(model):.2f} if 1.0 matches direct)")

print("\n[KEY] If SERVER_LR=1.0 doesn't match direct training, the delta math is wrong")

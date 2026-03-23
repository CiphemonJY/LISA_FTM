#!/usr/bin/env python3
"""Compare LoRA injection and ppl computation between quick_lr_test and debug6."""
import math, torch, torch.nn as nn, random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

MODEL_ID = "EleutherAI/pythia-70m"
LR = 8e-4
LORA_RANK, LORA_ALPHA = 4, 8.0
BATCH_SIZE, MAX_SEQ_LEN = 4, 64
TRAIN_BATCHES = 20
SEED = 42
MAX_TEST = 5

class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=8.0, dropout=0.05):
        super().__init__()
        self.weight_data = linear.weight.data.clone().float()
        self.bias_data = linear.bias.data.clone().float() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight_data.shape
        self.rank, self.alpha = rank, alpha
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

    def trainable_params(self):
        return [self.lora_A, self.lora_B]


class LoraWrapper:
    TARGET = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
              "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc"]

    def __init__(self, model, rank=4, alpha=8.0, dropout=0.05):
        self.model = model
        self.rank, self.alpha, self.dropout = rank, alpha, dropout
        self.lora_layers = {}

    def apply_lora(self):
        for full_name, module in self.model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv1d)):
                continue
            name_parts = full_name.split(".")
            if not any(tm in name_parts[-1] for tm in self.TARGET):
                continue
            lora = LoRALinear(module, rank=self.rank, alpha=self.alpha, dropout=self.dropout)
            self.lora_layers[full_name] = lora
            parts = full_name.rsplit(".", 1)
            if len(parts) == 2:
                try:
                    setattr(self.model.get_submodule(parts[0]), parts[1], lora)
                except KeyError:
                    pass
        return len(self.lora_layers)

    def freeze_all_A_only(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for lora_layer in self.lora_layers.values():
            lora_layer.lora_A.requires_grad = True
            lora_layer.lora_B.requires_grad = False

    def snapshot(self):
        return {f"{k}.A": l.lora_A.data.clone().cpu()
                     for k, l in self.lora_layers.items()} | \
               {f"{k}.B": l.lora_B.data.clone().cpu()
                     for k, l in self.lora_layers.items()}

    def restore(self, state):
        for k, l in self.lora_layers.items():
            l.lora_A.data.copy_(state[f"{k}.A"].clone())
            l.lora_B.data.copy_(state[f"{k}.B"].clone())


def tokenize(tok, texts):
    enc = tok(texts, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


def ppl_method_A(model, enc, tok):
    """debug6's ppl method: ignore_index = getattr(model.config, 'pad_token_id', None) or 0"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pad_id = getattr(model.config, "pad_token_id", None) or 0
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)
    for i in range(min(MAX_TEST, len(enc["input_ids"]) // BATCH_SIZE)):
        s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        ids = enc["input_ids"][s:e].clone()
        labs = enc["labels"][s:e].clone()
        with torch.no_grad():
            out = model(input_ids=ids)
        loss = crit(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1)), total_loss, total_tokens


def ppl_method_Q(model, enc, tok):
    """quick_lr_test's ppl method: ignore_index = tok.pad_token_id or 0"""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pad_id = tok.pad_token_id or 0
    crit = nn.CrossEntropyLoss(ignore_index=pad_id)
    for i in range(min(MAX_TEST, len(enc["input_ids"]) // BATCH_SIZE)):
        s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        ids = enc["input_ids"][s:e].clone()
        labs = enc["labels"][s:e].clone()
        with torch.no_grad():
            out = model(input_ids=ids)
        loss = crit(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1)), total_loss, total_tokens


print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
print(f"  tok.pad_token_id = {tok.pad_token_id}")

cfg = AutoConfig.from_pretrained(MODEL_ID)
print(f"  model.config.pad_token_id (before load) = None")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
print(f"  model.config.pad_token_id (after load) = {getattr(model.config, 'pad_token_id', 'NOT SET')}")

print("\nLoading wikitext...")
ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
ts = load_dataset("wikitext", "wikitext-2-v1", split="test")
train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
test_texts = [t for t in ts["text"] if t.strip() and len(t.strip()) > 20]
train_enc = tokenize(tok, train_texts)
test_enc = tokenize(tok, test_texts)

# BASE MODEL
print("\n=== BASE MODEL (no LoRA) ===")
ppl_A, loss_A, tokens_A = ppl_method_A(model, test_enc, tok)
ppl_Q, loss_Q, tokens_Q = ppl_method_Q(model, test_enc, tok)
print(f"  Method A (debug6):  ppl={ppl_A:.2f}, loss={loss_A:.4f}, tokens={tokens_A}")
print(f"  Method Q (quick):  ppl={ppl_Q:.2f}, loss={loss_Q:.4f}, tokens={tokens_Q}")

# APPLY LoRA
print("\n=== LoRA applied ===")
lm = LoraWrapper(model, rank=LORA_RANK, alpha=LORA_ALPHA)
n = lm.apply_lora()
print(f"  LoRA layers: {n}")
ppl_A, loss_A, tokens_A = ppl_method_A(model, test_enc, tok)
ppl_Q, loss_Q, tokens_Q = ppl_method_Q(model, test_enc, tok)
print(f"  Method A (debug6):  ppl={ppl_A:.2f}, loss={loss_A:.4f}, tokens={tokens_A}")
print(f"  Method Q (quick):  ppl={ppl_Q:.2f}, loss={loss_Q:.4f}, tokens={tokens_Q}")

# Check: what are the actual logits?
print("\n=== Logit check ===")
model.eval()
ids = test_enc["input_ids"][:2].clone()
labs = test_enc["labels"][:2].clone()
with torch.no_grad():
    out = model(input_ids=ids)
logits = out.logits
print(f"  logits shape: {logits.shape}")
print(f"  logits min: {logits.min():.4f}, max: {logits.max():.4f}")
print(f"  logits has NaN: {torch.isnan(logits).any()}")
print(f"  logits has Inf: {torch.isinf(logits).any()}")

# Check what pad tokens are
print(f"\n  pad token id: {tok.pad_token_id}")
print(f"  input_ids[0, :10]: {ids[0, :10].tolist()}")
print(f"  labels[0, :10]:   {labs[0, :10].tolist()}")
print(f"  number of pad tokens in ids[0]: {(ids[0] == tok.pad_token_id).sum().item()}")

# Compare with EOS token
print(f"\n  eos token id: {tok.eos_token_id}")
print(f"  input_ids[0, -10:]: {ids[0, -10:].tolist()}")
print(f"  number of eos tokens in ids[0]: {(ids[0] == tok.eos_token_id).sum().item()}")

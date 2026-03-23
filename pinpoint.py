#!/usr/bin/env python3
"""Pinpoint test: 1 client, 1 batch. See exactly what training does to LoRA weights."""
import math, torch, torch.nn as nn, random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

MODEL_ID = "EleutherAI/pythia-70m"
LR = 8e-4
LORA_RANK, LORA_ALPHA = 4, 8.0
BATCH_SIZE, MAX_SEQ_LEN = 4, 64
SEED = 42
SERVER_LR = 0.1

random.seed(SEED); torch.manual_seed(SEED)

# ── LoRA (exact copy from byzantine_stress_test.py) ────────────────────────
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

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for lora_layer in self.lora_layers.values():
            lora_layer.lora_A.requires_grad = True
            lora_layer.lora_B.requires_grad = True

    def freeze_all_A_only(self):
        for p in self.model.parameters():
            p.requires_grad = False
        for lora_layer in self.lora_layers.values():
            lora_layer.lora_A.requires_grad = True
            lora_layer.lora_B.requires_grad = False

    def unfreeze_lora_layers(self, layer_indices):
        patterns = []
        for idx in layer_indices:
            patterns.extend([f"gpt_neox.layers.{idx}.", f".h.{idx}."])
        for full_name, lora_layer in self.lora_layers.items():
            for pat in patterns:
                if pat in full_name:
                    lora_layer.lora_A.requires_grad = True
                    lora_layer.lora_B.requires_grad = True
                    break

    def snapshot(self):
        return {k: (l.lora_A.data.clone(), l.lora_B.data.clone())
                for k, l in self.lora_layers.items()}

    def restore(self, state):
        for k, l in self.lora_layers.items():
            a, b = state[k]
            l.lora_A.data.copy_(a); l.lora_B.data.copy_(b)


# ── Data ─────────────────────────────────────────────────────────────────────
print("Loading model and tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
print(f"  LoRA wrapper created")

print("Loading wikitext...")
ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
random.shuffle(texts)
client_texts = texts[:200]  # small for fast test

enc = tok(client_texts, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
enc["labels"] = enc["input_ids"].clone()
print(f"  {len(client_texts)} texts, batch shape: {enc['input_ids'].shape}")

# ── Apply LoRA ───────────────────────────────────────────────────────────────
lm = LoraWrapper(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.05)
n = lm.apply_lora()
print(f"\nLoRA applied to {n} layers")

# ── PPL before training ──────────────────────────────────────────────────────
def compute_ppl(m, enc_data, n_batches=3):
    m.eval()
    total_loss, total_tokens = 0.0, 0
    crit = nn.CrossEntropyLoss(ignore_index=0)
    for i in range(min(n_batches, len(enc_data["input_ids"]) // BATCH_SIZE)):
        s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        ids = enc_data["input_ids"][s:e].clone()
        labs = enc_data["labels"][s:e].clone()
        with torch.no_grad():
            out = m(input_ids=ids)
        loss = crit(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    m.train()
    return math.exp(total_loss / max(total_tokens, 1))

ppl_before = compute_ppl(model, enc)
print(f"\nPPL before training: {ppl_before:.2f}")

# ── Snapshot before ──────────────────────────────────────────────────────────
snap_before = lm.snapshot()

# ── Train ONE batch (A-only unfreeze) ────────────────────────────────────────
print(f"\n=== Training 1 batch (freeze A-only unfreeze) ===")
lm.freeze_all_A_only()

opt = torch.optim.Adam([p for l in lm.lora_layers.values() for p in l.trainable_params()], lr=LR)
crit = nn.CrossEntropyLoss(ignore_index=0)

ids = enc["input_ids"][:BATCH_SIZE].clone()
labs = enc["labels"][:BATCH_SIZE].clone()
opt.zero_grad()
out = model(input_ids=ids)
loss = crit(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
loss.backward()
opt.step()

snap_after_A = lm.snapshot()
print(f"  loss: {loss.item():.4f}")

# ── Train 19 more batches (same setup) ─────────────────────────────────────
print(f"\n=== Training 19 more batches (total 20) ===")
for batch_i in range(1, 20):
    ids = enc["input_ids"][batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE].clone()
    labs = enc["labels"][batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE].clone()
    opt.zero_grad()
    out = model(input_ids=ids)
    loss = crit(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
    loss.backward()
    opt.step()
print(f"  final loss: {loss.item():.4f}")

snap_after_20 = lm.snapshot()

# ── Show weight deltas ───────────────────────────────────────────────────────
print(f"\n=== LoRA weight changes after 20 batches ===")
key = list(lm.lora_layers.keys())[0]
for k in list(lm.lora_layers.keys())[:4]:
    a_before = snap_before[k][0].norm().item()
    a_after = snap_after_20[k][0].norm().item()
    b_before = snap_before[k][1].norm().item()
    b_after = snap_after_20[k][1].norm().item()
    print(f"  {k.split('.')[-1]}: A {a_before:.6f} -> {a_after:.6f} (dA={a_after-a_before:+.6f})")
    print(f"                        B {b_before:.6f} -> {b_after:.6f} (dB={b_after-b_before:+.6f})")

# ── Compute gradient delta norms ──────────────────────────────────────────────
print(f"\n=== Computing gradient delta (W_after - W_before) ===")
grads = {}
for k, l in lm.lora_layers.items():
    da = snap_after_20[k][0] - snap_before[k][0]
    db = snap_after_20[k][1] - snap_before[k][1]
    grads[k] = {"A": da, "B": db}
    print(f"  {k.split('.')[-1]}: ||dA|| = {da.norm().item():.6f}, ||dB|| = {db.norm().item():.6f}")

# ── Apply delta with SERVER_LR ───────────────────────────────────────────────
print(f"\n=== Simulating FedAvg aggregation (SERVER_LR={SERVER_LR}) ===")
# Restore original, then apply delta
lm.restore(snap_before)

key = list(lm.lora_layers.keys())[0]
for k, l in lm.lora_layers.items():
    l.lora_A.data.add_(grads[k]["A"], alpha=SERVER_LR)
    l.lora_B.data.add_(grads[k]["B"], alpha=SERVER_LR)

snap_agg = lm.snapshot()
ppl_after_agg = compute_ppl(model, enc)
print(f"\nPPL after aggregation: {ppl_after_agg:.2f}")
print(f"PPL change: {ppl_before:.2f} -> {ppl_after_agg:.2f} ({ppl_after_agg/ppl_before:.2f}x)")

# ── What if we DON'T apply any delta? ───────────────────────────────────────
lm.restore(snap_before)
ppl_no_delta = compute_ppl(model, enc)
print(f"\nPPL without applying delta: {ppl_no_delta:.2f}")

# ── What if we apply only A delta? ──────────────────────────────────────────
lm.restore(snap_before)
for k, l in lm.lora_layers.items():
    l.lora_A.data.add_(grads[k]["A"], alpha=SERVER_LR)
ppl_A_only = compute_ppl(model, enc)
print(f"PPL with only A delta (SERVER_LR={SERVER_LR}): {ppl_A_only:.2f}")

# ── What if we apply only B delta? ──────────────────────────────────────────
lm.restore(snap_before)
for k, l in lm.lora_layers.items():
    l.lora_B.data.add_(grads[k]["B"], alpha=SERVER_LR)
ppl_B_only = compute_ppl(model, enc)
print(f"PPL with only B delta (SERVER_LR={SERVER_LR}): {ppl_B_only:.2f}")

# ── What if SERVER_LR is tiny? ───────────────────────────────────────────────
lm.restore(snap_before)
for k, l in lm.lora_layers.items():
    l.lora_A.data.add_(grads[k]["A"], alpha=0.0001)
    l.lora_B.data.add_(grads[k]["B"], alpha=0.0001)
ppl_tiny = compute_ppl(model, enc)
print(f"PPL with SERVER_LR=0.0001: {ppl_tiny:.2f}")

#!/usr/bin/env python3
"""Debug why LoRA gradients are zero."""
import math, torch, torch.nn as nn, random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

MODEL_ID = "EleutherAI/pythia-70m"
LORA_RANK, LORA_ALPHA = 4, 8.0
BATCH_SIZE, MAX_SEQ_LEN = 4, 64
LR = 8e-4

random.seed(42); torch.manual_seed(42)

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
        return {k: (l.lora_A.data.clone(), l.lora_B.data.clone())
                for k, l in self.lora_layers.items()}


print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)

ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20][:100]
enc = tok(texts, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
enc["labels"] = enc["input_ids"].clone()

lm = LoraWrapper(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=0.05)
n = lm.apply_lora()
print(f"LoRA applied to {n} layers")

# Check what's trainable
print("\n=== Trainable params before freeze ===")
for name, p in model.named_parameters():
    if p.requires_grad:
        print(f"  {name}: shape={p.shape}, norm={p.norm().item():.6f}")

lm.freeze_all_A_only()

print("\n=== Trainable params after freeze_all_A_only ===")
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
print(f"  Total trainable: {len(trainable)}")
for name, p in trainable[:5]:
    print(f"  {name}: shape={p.shape}, norm={p.norm().item():.6f}")

# Collect optimizer params
opt_params = [p for n, p in model.named_parameters() if p.requires_grad]
print(f"\n=== Optimizer params ===")
print(f"  Count: {len(opt_params)}")
if opt_params:
    print(f"  First: {trainable[0][0]} shape={opt_params[0].shape}")
    print(f"  Are they lora_A? {all('lora' in n for n,p in trainable)}")

# Training step with full gradient tracking
print("\n=== Training step with gradient check ===")
opt = torch.optim.Adam(opt_params, lr=LR)
crit = nn.CrossEntropyLoss(ignore_index=0)

ids = enc["input_ids"][:BATCH_SIZE].clone()
labs = enc["labels"][:BATCH_SIZE].clone()

# Check lora_A grads BEFORE step
opt.zero_grad()

# Verify lora_A is in model.parameters()
key = list(lm.lora_layers.keys())[0]
la_param = lm.lora_layers[key].lora_A
print(f"  lora_A id in model: {any(p is la_param for p in model.parameters())}")
print(f"  lora_A requires_grad: {la_param.requires_grad}")
print(f"  lora_A is in opt params: {any(p is la_param for p in opt_params)}")

out = model(input_ids=ids)
loss = crit(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
print(f"  loss: {loss.item():.4f}")

# Before backward
la_grad_before = la_param.grad.clone() if la_param.grad is not None else None
print(f"  lora_A grad before backward: {la_grad_before.norm().item() if la_grad_before is not None else 'None'}")

loss.backward()

la_grad_after = la_param.grad
print(f"  lora_A grad after backward: {la_grad_after.norm().item() if la_grad_after is not None else 'None'}")

# Check all grads
print(f"\n  All trainable grads:")
for name, p in trainable:
    gnorm = p.grad.norm().item() if p.grad is not None else 0
    print(f"    {name}: grad_norm={gnorm:.8f}")

opt.step()

# Check weights after
snap = lm.snapshot()
key = list(lm.lora_layers.keys())[0]
a_after = snap[key][0].norm().item()
print(f"\n  lora_A norm after step: {a_after:.8f}")
print(f"  lora_A changed: {abs(a_after - 0.447185) > 1e-8}")

#!/usr/bin/env python3
"""Quick test: find the right SERVER_LR that prevents divergence."""
import sys, os, math, random, gc
sys.path.insert(0, os.path.dirname(__file__))

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL_ID = "EleutherAI/pythia-70m"
LR = 8e-4
LORA_RANK, LORA_ALPHA = 4, 8.0
BATCH_SIZE, MAX_SEQ_LEN = 4, 64
TRAIN_BATCHES = 20
NUM_CLIENTS = 3
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

    def unfreeze_A_only(self, layer_indices):
        for full_name, lora_layer in self.lora_layers.items():
            if any(f"layers.{idx}." in full_name for idx in layer_indices):
                lora_layer.lora_A.requires_grad = True
                lora_layer.lora_B.requires_grad = False

    def snapshot(self):
        return {f"{k}.A": l.lora_A.data.clone().cpu() for k, l in self.lora_layers.items()} | \
               {f"{k}.B": l.lora_B.data.clone().cpu() for k, l in self.lora_layers.items()}

    def restore(self, state):
        for k, l in self.lora_layers.items():
            l.lora_A.data.copy_(state[f"{k}.A"].clone())
            l.lora_B.data.copy_(state[f"{k}.B"].clone())


def tokenize(tok, texts):
    enc = tok(texts, max_length=MAX_SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


def ppl(model, enc, tok):
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
    return math.exp(total_loss / max(total_tokens, 1))


# Load model
print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
lm = LoraWrapper(model, rank=LORA_RANK, alpha=LORA_ALPHA)
n = lm.apply_lora()
print(f"  LoRA={n}, pad_token_id={tok.pad_token_id}")

# Load data
print("Loading wikitext...")
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
ts = load_dataset("wikitext", "wikitext-2-v1", split="test")
train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
test_texts = [t for t in ts["text"] if t.strip() and len(t.strip()) > 20]
train_enc = tokenize(tok, train_texts)
test_enc = tokenize(tok, test_texts)

# Partition
rng = random.Random(SEED)
shuffled = list(train_texts)
rng.shuffle(shuffled)
n_per = len(shuffled) // NUM_CLIENTS
client_encs = [tokenize(tok, shuffled[i*n_per:(i+1)*n_per if i < NUM_CLIENTS-1 else len(shuffled)])
               for i in range(NUM_CLIENTS)]
counts = [len(c["input_ids"]) for c in client_encs]
print(f"  texts per client: {counts}")


def run_one_round(srv_lr):
    """Run one federated round, return ppl_after."""
    # Fresh LoRA
    for p in model.parameters():
        p.requires_grad = False
    for lora_layer in lm.lora_layers.values():
        lora_layer.lora_A.data.normal_(mean=0, std=0.01)
        lora_layer.lora_B.data.zero_()

    ppl_before = ppl(model, test_enc, tok)
    snap = lm.snapshot()

    deltas, weights = [], []
    for ci in range(NUM_CLIENTS):
        lm.freeze_all_A_only()
        lm.unfreeze_A_only(list(range(6)))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

        for _ in range(TRAIN_BATCHES):
            idx = torch.randperm(len(client_encs[ci]["input_ids"]))[:BATCH_SIZE].tolist()
            ids = client_encs[ci]["input_ids"][idx].clone().clamp(0, tok.vocab_size - 1)
            labs = client_encs[ci]["labels"][idx].clone().clamp(0, tok.vocab_size - 1)
            opt.zero_grad()
            out = model(input_ids=ids, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        weights.append(len(client_encs[ci]["input_ids"]))
        snap_after = lm.snapshot()
        deltas.append({k: snap_after[k] - snap[k] for k in snap})
        lm.restore(snap)

    # Aggregate
    total_w = sum(weights)
    nw = [w / total_w for w in weights]
    acc = {}
    for delta, w in zip(deltas, nw):
        for k, v in delta.items():
            acc[k] = acc.get(k, torch.zeros_like(v)) + v.float() * w

    delta_norms = [v.norm().item() for v in acc.values()]
    avg_norm = sum(delta_norms) / len(delta_norms)
    max_norm = max(delta_norms)

    # Apply
    for k, l in lm.lora_layers.items():
        with torch.no_grad():
            l.lora_A.data.add_(acc[f"{k}.A"], alpha=srv_lr)
            l.lora_B.data.add_(acc[f"{k}.B"], alpha=srv_lr)

    ppl_after = ppl(model, test_enc, tok)
    return ppl_before, ppl_after, avg_norm, max_norm


print(f"\n--- Testing SERVER_LR (1 round, {NUM_CLIENTS} clients, {TRAIN_BATCHES} batches each) ---")
print(f"  LR={LR}, LORA_RANK={LORA_RANK}, ALPHA={LORA_ALPHA}")
print()

for srv_lr in [1.0, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]:
    ppl_before, ppl_after, avg_norm, max_norm = run_one_round(srv_lr)
    status = "DIVERGED" if ppl_after > 1e10 else "OK"
    print(f"  SERVER_LR={srv_lr:>7}: acc_norm_avg={avg_norm:.4f} max={max_norm:.4f} | ppl {ppl_before:.0f} -> {ppl_after:.0f} [{status}]")

print("\nDone.")

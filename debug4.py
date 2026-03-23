#!/usr/bin/env python3
"""Simulate the exact multi-client, multi-round byzantine loop to find divergence source."""
import sys, os, math, random, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---------------------------------------------------------------------------
# Config (define FIRST so all functions can reference them)
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/pythia-70m"
NUM_CLIENTS = 3
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1
LR = 8e-4
LORA_RANK = 4
LORA_ALPHA = 8.0
LISA_BOTTOM = 2
LISA_TOP = 2
LISA_MIDDLE = 2
NUM_LAYERS = 6
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
MAX_TRAIN_BATCHES = 20
SEED = 42
NUM_TEST_BATCHES = 10
SERVER_LR = 1.0

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=8.0, dropout=0.05):
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
        lora = nn.functional.linear(self.lora_dropout(x_f32), self.lora_A)
        lora = nn.functional.linear(lora, self.lora_B)
        return (original + lora * self.scaling).to(orig_dtype)

    def trainable_params(self):
        return [self.lora_A, self.lora_B]


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
        return count

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_lora_layers(self, layer_indices):
        patterns = []
        for idx in layer_indices:
            patterns.extend([f"gpt_neox.layers.{idx}.", f".h.{idx}."])
        for full_name, lora_layer in self.lora_layers.items():
            for pat in patterns:
                if pat in full_name:
                    for p in lora_layer.trainable_params():
                        p.requires_grad = True
                    break

    def snapshot(self):
        return {f"{k}.A": l.lora_A.data.clone().cpu() for k, l in self.lora_layers.items()} | \
               {f"{k}.B": l.lora_B.data.clone().cpu() for k, l in self.lora_layers.items()}

    def restore(self, state):
        for k, l in self.lora_layers.items():
            l.lora_A.data.copy_(state[f"{k}.A"].clone())
            l.lora_B.data.copy_(state[f"{k}.B"].clone())

    def train_client(self, enc, tok, selected, lr, n_batches):
        self.freeze_all()
        self.unfreeze_lora_layers(selected)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        losses = []
        for _ in range(n_batches):
            idx = torch.randperm(len(enc["input_ids"]))[:BATCH_SIZE].tolist()
            ids = enc["input_ids"][idx].clone().clamp(0, tok.vocab_size - 1)
            labs = enc["labels"][idx].clone().clamp(0, tok.vocab_size - 1)
            opt.zero_grad()
            out = self.model(input_ids=ids, labels=labs)
            loss = out.loss
            if torch.isnan(loss):
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)


def compute_deltas(before, after):
    return {k: after[k] - before[k] for k in before}


def aggregate_deltas(deltas, weights, wrapper, srv_lr=1.0):
    acc = {}
    for delta, w in zip(deltas, weights):
        for k, v in delta.items():
            acc[k] = acc.get(k, torch.zeros_like(v.float())) + v.float() * w
    total_w = sum(weights)
    for k in acc:
        acc[k] /= total_w
    for k, l in wrapper.lora_layers.items():
        l.lora_A.data.add_(acc[f"{k}.A"], alpha=srv_lr)
        l.lora_B.data.add_(acc[f"{k}.B"], alpha=srv_lr)


def ppl(model, enc, tok):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    for i in range(min(NUM_TEST_BATCHES, len(enc["input_ids"]) // BATCH_SIZE)):
        s, e = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
        ids = enc["input_ids"][s:e].clone()
        labs = enc["labels"][s:e].clone()
        with torch.no_grad():
            out = model(input_ids=ids)
        loss = criterion(out.logits.view(-1, out.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def lisa_select(num_layers, bottom=LISA_BOTTOM, top=LISA_TOP, middle=LISA_MIDDLE, seed=None):
    rng = random.Random(seed) if seed is not None else random
    b = list(range(min(bottom, num_layers)))
    t = list(range(max(0, num_layers - top), num_layers))
    mp = list(range(bottom, max(bottom, num_layers - top)))
    m = rng.sample(mp, min(middle, len(mp))) if mp else []
    return sorted(set(b + t + m))


def tokenize(tok, texts):
    enc = tok(texts, max_length=MAX_SEQ_LEN, padding="max_length",
              truncation=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg,
             trust_remote_code=True, torch_dtype=torch.float32)
wrapper = LoraAppliedModel(model, rank=LORA_RANK, alpha=LORA_ALPHA)
count = wrapper.apply_lora()
print(f"  LoRA: {count} layers, pad_token_id={tok.pad_token_id}")

print("Loading data...")
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    ts = load_dataset("wikitext", "wikitext-2-v1", split="test")
    train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
    test_texts = [t for t in ts["text"] if t.strip() and len(t.strip()) > 20]
except Exception as e:
    print(f"  synthetic: {e}")
    train_texts = [" ".join(["word"]*30) for _ in range(600)]
    test_texts = [" ".join(["word"]*30) for _ in range(50)]

train_enc = tokenize(tok, train_texts)
test_enc = tokenize(tok, test_texts)

# Partition
rng = random.Random(SEED)
shuffled = list(train_texts)
rng.shuffle(shuffled)
n = len(shuffled) // NUM_CLIENTS
client_texts = [shuffled[i*n:(i+1)*n if i < NUM_CLIENTS-1 else len(shuffled)]
                for i in range(NUM_CLIENTS)]
client_encs = [tokenize(tok, ct) for ct in client_texts]
print(f"  {len(train_texts)} texts, {[len(c) for c in client_texts]} per client")

print(f"\n--- {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients, LR={LR}, SERVER_LR={SERVER_LR} ---")

for r in range(1, NUM_ROUNDS+1):
    print(f"\n--- Round {r} ---")
    snap = wrapper.snapshot()
    deltas = []
    weights = []
    losses = []

    for ci in range(NUM_CLIENTS):
        sel = lisa_select(NUM_LAYERS, seed=(r * 100 + ci * 17 + SEED))
        loss = wrapper.train_client(client_encs[ci], tok, sel, LR, MAX_TRAIN_BATCHES)
        if loss is None:
            print(f"  Client {ci}: NaN!"); break
        losses.append(loss)
        weights.append(len(client_texts[ci]))
        snap_after = wrapper.snapshot()
        deltas.append(compute_deltas(snap, snap_after))
        wrapper.restore(snap)
        print(f"  C{ci}: loss={loss:.4f}, sel={sel}")

    if len(losses) < NUM_CLIENTS:
        print("  Aborting round due to NaN")
        break

    ppl_before = ppl(model, test_enc, tok)
    aggregate_deltas(deltas, weights, wrapper, srv_lr=SERVER_LR)
    ppl_after = ppl(model, test_enc, tok)
    print(f"  R{r}: ppl={ppl_after:.2f} (delta={ppl_after-ppl_before:+.2f}), avg_loss={sum(losses)/len(losses):.4f}")
    gc.collect()

print("\nDone.")

#!/usr/bin/env python3
"""Pinpoint why byzantine_stress_test.py diverges: instrument the aggregation."""
import sys, os, math, random, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/pythia-70m"
LR = 8e-4
LORA_RANK, LORA_ALPHA = 4, 8.0
LISA_BOTTOM, LISA_TOP, LISA_MIDDLE = 2, 2, 2
NUM_LAYERS = 6
BATCH_SIZE, MAX_SEQ_LEN = 4, 128
TRAIN_BATCHES, MAX_TEST = 1, 10  # 1 epoch = ~1728 batches per client (byzantine uses n_batches = min(1728, MAX_TRAIN_BATCHES_PER_CLIENT))
NUM_CLIENTS = 3
NUM_ROUNDS = 3
SEED = 42
SERVER_LR = 0.1   # test with 0.1

# ---------------------------------------------------------------------------
# LoRA (exact copy from byzantine_stress_test.py)
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, alpha=8.0, dropout=0.05):
        super().__init__()
        self.weight_data = linear.weight.data.clone().float()
        self.bias_data = linear.bias.data.clone().float() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight_data.shape
        self.is_conv1d = isinstance(linear, nn.Conv1d)
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))  # lora_B=False
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
                parent_name, attr = parts
                try:
                    setattr(self.model.get_submodule(parent_name), attr, lora)
                except KeyError:
                    pass
        return len(self.lora_layers)

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        with torch.no_grad():
            original = nn.functional.linear(x_f32, self.weight_data, self.bias_data)
        lora = nn.functional.linear(self.lora_dropout(x_f32), self.lora_A)
        lora = nn.functional.linear(lora, self.lora_B)
        return (original + lora * self.scaling).to(orig_dtype)

    def trainable_params(self):
        return [self.lora_A, self.lora_B]  # only lora_A trains (B starts at zero, doesn't train)

    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

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

    def freeze_all_A_only(self):
        """Freeze everything, then unfreeze only lora_A (not lora_B) for selected layers."""
        for p in self.model.parameters():
            p.requires_grad = False
        for lora_layer in self.lora_layers.values():
            for p in lora_layer.trainable_params():
                p.requires_grad = False
            # Keep lora_A requires_grad as-is (default False, will be set per client)

    def unfreeze_lora_A_only(self, layer_indices):
        """Unfreeze only lora_A (not lora_B) for layers matching indices."""
        patterns = []
        for idx in layer_indices:
            patterns.extend([f"gpt_neox.layers.{idx}.", f".h.{idx}."])
        for full_name, lora_layer in self.lora_layers.items():
            for pat in patterns:
                if pat in full_name:
                    lora_layer.lora_A.requires_grad = True
                    lora_layer.lora_B.requires_grad = False
                    break

    def snapshot(self):
        return {f"{k}.A": l.lora_A.data.clone().cpu()
                     for k, l in self.lora_layers.items()} | \
               {f"{k}.B": l.lora_B.data.clone().cpu()
                     for k, l in self.lora_layers.items()}

    def restore(self, state):
        for k, l in self.lora_layers.items():
            l.lora_A.data.copy_(state[f"{k}.A"].clone())
            l.lora_B.data.copy_(state[f"{k}.B"].clone())

    def apply_delta(self, delta, alpha):
        for k, l in self.lora_layers.items():
            l.lora_A.data.add_(delta[f"{k}.A"], alpha=alpha)
            l.lora_B.data.add_(delta[f"{k}.B"], alpha=alpha)

    def count_trainable(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tokenize(tok, texts):
    enc = tok(texts, max_length=MAX_SEQ_LEN, padding="max_length",
              truncation=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc

def ppl(model, enc, tok):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
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

def lisa_select(num_layers, bottom=LISA_BOTTOM, top=LISA_TOP, middle=LISA_MIDDLE, seed=None):
    rng = random.Random(seed) if seed is not None else random
    b = list(range(min(bottom, num_layers)))
    t = list(range(max(0, num_layers - top), num_layers))
    mp = list(range(bottom, max(bottom, num_layers - top)))
    m = rng.sample(mp, min(middle, len(mp))) if mp else []
    return sorted(set(b + t + m))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg, trust_remote_code=True, torch_dtype=torch.float32)
lm = LoraAppliedModel(model, rank=LORA_RANK, alpha=LORA_ALPHA)
n_lora = lm.apply_lora()
print(f"  LoRA: {n_lora} layers, pad_token_id={tok.pad_token_id}")

print("Loading wikitext...")
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    ts = load_dataset("wikitext", "wikitext-2-v1", split="test")
    train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
    test_texts = [t for t in ts["text"] if t.strip() and len(t.strip()) > 20]
except:
    train_texts = [" ".join(["word"]*30) for _ in range(600)]
    test_texts = [" ".join(["word"]*30) for _ in range(50)]

train_enc = tokenize(tok, train_texts)
test_enc = tokenize(tok, test_texts)

# Partition
rng = random.Random(SEED)
shuffled = list(train_texts)
rng.shuffle(shuffled)
n = len(shuffled) // NUM_CLIENTS
client_texts = [shuffled[i*n:(i+1)*n if i < NUM_CLIENTS-1 else len(shuffled)] for i in range(NUM_CLIENTS)]
client_encs = [tokenize(tok, ct) for ct in client_texts]
print(f"  {[len(c) for c in client_texts]} texts/client")

# ---- Instrumented federated loop ----
print(f"\n--- Instrumented loop: LR={LR}, SERVER_LR={SERVER_LR} ---")
lm.freeze_all()

for r in range(1, NUM_ROUNDS+1):
    print(f"\n=== Round {r} ===")
    snap = lm.snapshot()

    # Check: are snapshot values valid?
    sample_key = list(snap.keys())[0]
    print(f"  snap[{sample_key[:40]}...] norm={snap[sample_key].norm():.6f}")

    ppl_before = ppl(model, test_enc, tok)
    print(f"  ppl before round: {ppl_before:.2f}")

    deltas, weights, losses = [], [], []
    for ci in range(NUM_CLIENTS):
        sel = lisa_select(NUM_LAYERS, seed=(r * 100 + ci * 17 + SEED))
        lm.freeze_all_A_only()
        lm.unfreeze_lora_A_only(sel)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)

        batch_losses = []
        for _ in range(TRAIN_BATCHES):
            idx = torch.randperm(len(client_encs[ci]["input_ids"]))[:BATCH_SIZE].tolist()
            ids = client_encs[ci]["input_ids"][idx].clone().clamp(0, tok.vocab_size - 1)
            labs = client_encs[ci]["labels"][idx].clone().clamp(0, tok.vocab_size - 1)
            opt.zero_grad()
            out = model(input_ids=ids, labels=labs)
            loss = out.loss
            if torch.isnan(loss):
                batch_losses.append(float('nan'))
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else float('nan')
        losses.append(avg_loss)
        weights.append(len(client_texts[ci]))

        snap_after = lm.snapshot()
        delta = {k: snap_after[k] - snap[k] for k in snap}
        deltas.append(delta)

        delta_A_norms = [v.norm().item() for k, v in delta.items() if '.A' in k]
        delta_B_norms = [v.norm().item() for k, v in delta.items() if '.B' in k]
        print(f"  C{ci}: loss={avg_loss:.4f}, delta_A_norm={sum(delta_A_norms)/len(delta_A_norms):.6f}, delta_B_norm={sum(delta_B_norms)/len(delta_B_norms):.6f}")

        lm.restore(snap)

    # ---- Aggregate ----
    # Exact copy from byzantine_stress_test.py
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    deltas_copy = [{k: v.float() for k, v in d.items()} for d in deltas]
    acc = {}
    for delta, w in zip(deltas_copy, norm_weights):
        for k, v in delta.items():
            acc[k] = acc.get(k, torch.zeros_like(v)) + v * w

    # Check: what's the magnitude of acc?
    acc_norms = [v.norm().item() for v in acc.values()]
    print(f"  [AGGREGATE] acc norm avg={sum(acc_norms)/len(acc_norms):.6f} max={max(acc_norms):.6f}")

    # Apply with SERVER_LR
    for k, l in lm.lora_layers.items():
        a_key, b_key = f"{k}.A", f"{k}.B"
        if a_key in acc and b_key in acc:
            with torch.no_grad():
                l.lora_A.data.add_(acc[a_key] * SERVER_LR)
                l.lora_B.data.add_(acc[b_key] * SERVER_LR)

    ppl_after = ppl(model, test_enc, tok)
    print(f"  R{r}: ppl={ppl_after:.2f} (delta={ppl_after-ppl_before:+.2f}), avg_train_loss={sum(losses)/len(losses):.4f}")

    if ppl_after > 1e10:
        print("  !!! CATASTROPHIC DIVERGENCE - checking LoRA weights")
        sample_lora = list(lm.lora_layers.values())[0]
        print(f"  lora_A norm: {sample_lora.lora_A.data.norm():.4f}")
        print(f"  lora_B norm: {sample_lora.lora_B.data.norm():.4f}")
        # Check logits
        model.eval()
        ids = test_enc["input_ids"][:2].clone()
        with torch.no_grad():
            out = model(input_ids=ids)
            logits = out.logits
            print(f"  logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            print(f"  logits has NaN: {torch.isnan(logits).any()}")
            print(f"  logits has Inf: {torch.isinf(logits).any()}")
        model.train()
        break

    gc.collect()

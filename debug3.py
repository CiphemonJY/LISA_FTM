#!/usr/bin/env python3
"""Simulate byzantine_stress_test.py client loop exactly to find why it diverges."""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---------------------------------------------------------------------------
# LoRA — identical to byzantine_stress_test.py
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
        print(f"  LoRA applied to {count} layers")
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
        state = {}
        for full_name, lora_layer in self.lora_layers.items():
            state[f"{full_name}.lora_A"] = lora_layer.lora_A.data.clone().cpu()
            state[f"{full_name}.lora_B"] = lora_layer.lora_B.data.clone().cpu()
        return state

    def restore(self, state):
        for full_name, lora_layer in self.lora_layers.items():
            lora_layer.lora_A.data.copy_(state[f"{full_name}.lora_A"].clone())
            lora_layer.lora_B.data.copy_(state[f"{full_name}.lora_B"].clone())


# ---------------------------------------------------------------------------
# Config (same as byzantine_stress_test.py)
# ---------------------------------------------------------------------------
MODEL_ID = "EleutherAI/pythia-70m"
LR = 8e-4
LORA_RANK = 4
LORA_ALPHA = 8.0
LISA_BOTTOM = 2
LISA_TOP = 2
LISA_MIDDLE = 2
NUM_LAYERS = 6
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
TRAIN_BATCHES = 20
MAX_TEST_BATCHES = 10
SEED = 42


def tokenize(tokenizer, texts):
    enc = tokenizer(texts, max_length=MAX_SEQ_LEN, padding="max_length",
                    truncation=True, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    return enc


def ppl(model, test_enc, tokenizer):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    pad_id = getattr(tokenizer, 'pad_token_id', None) or 0
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


def lisa_select_layers(num_layers, bottom=LISA_BOTTOM, top=LISA_TOP, middle=LISA_MIDDLE, seed=None):
    rng = random.Random(seed) if seed is not None else random
    bottom_set = list(range(min(bottom, num_layers)))
    top_set = list(range(max(0, num_layers - top), num_layers))
    middle_pool = list(range(bottom, max(bottom, num_layers - top)))
    middle_set = rng.sample(middle_pool, min(middle, len(middle_pool))) if middle_pool else []
    return sorted(set(bottom_set + top_set + middle_set))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
cfg = AutoConfig.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=cfg,
             trust_remote_code=True, torch_dtype=torch.float32)
print(f"  pad_token_id={tok.pad_token_id}, config.pad_token_id={getattr(cfg, 'pad_token_id', None)}")

wrapper = LoraAppliedModel(model, rank=LORA_RANK, alpha=LORA_ALPHA)
wrapper.apply_lora()
wrapper.freeze_all()

print("Loading wikitext...")
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    ts = load_dataset("wikitext", "wikitext-2-v1", split="test")
    train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
    test_texts = [t for t in ts["text"] if t.strip() and len(t.strip()) > 20]
    print(f"  {len(train_texts)} train, {len(test_texts)} test")
except Exception as e:
    print(f"  synthetic: {e}")
    train_texts = [" ".join(["word"]*30) for _ in range(200)]
    test_texts = [" ".join(["word"]*30) for _ in range(50)]

train_enc = tokenize(tok, train_texts)
test_enc = tokenize(tok, test_texts)

print(f"\nConfig: LR={LR}, LISA bottom={LISA_BOTTOM}, top={LISA_TOP}, middle={LISA_MIDDLE}")

selected = lisa_select_layers(NUM_LAYERS, seed=SEED)
print(f"Selected layers: {selected}")

ppl_before = ppl(model, test_enc, tok)
print(f"ppl before: {ppl_before:.2f}")

snap = wrapper.snapshot()
wrapper.freeze_all()
wrapper.unfreeze_lora_layers(selected)
trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

opt = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
losses = []
for i in range(TRAIN_BATCHES):
    idx = torch.randperm(len(train_enc["input_ids"]))[:BATCH_SIZE].tolist()
    ids = train_enc["input_ids"][idx].clone().clamp(0, tok.vocab_size - 1)
    labs = train_enc["labels"][idx].clone().clamp(0, tok.vocab_size - 1)
    opt.zero_grad()
    out = model(input_ids=ids, labels=labs)
    loss = out.loss
    if torch.isnan(loss):
        print(f"  NaN at step {i}!")
        break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    opt.step()
    losses.append(loss.item())
    if i % 5 == 0:
        print(f"  step {i}: loss={loss.item():.4f}")

print(f"  avg train loss: {sum(losses)/len(losses):.4f}")
ppl_after_train = ppl(model, test_enc, tok)
print(f"ppl after training: {ppl_after_train:.2f}")

snap_after = wrapper.snapshot()
delta = {k: snap_after[k] - snap[k] for k in snap}
delta_norms = [v.norm().item() for v in delta.values()]
print(f"Delta norms: avg={sum(delta_norms)/len(delta_norms):.6f} max={max(delta_norms):.6f}")

# Apply delta with SERVER_LR=1.0
wrapper.restore(snap)
for full_name, lora_layer in wrapper.lora_layers.items():
    lora_layer.lora_A.data.add_(delta[f"{full_name}.lora_A"])
    lora_layer.lora_B.data.add_(delta[f"{full_name}.lora_B"])

ppl_after_delta = ppl(model, test_enc, tok)
print(f"ppl after delta (SERVER_LR=1.0): {ppl_after_delta:.2f}")
print(f"  vs direct train delta: {ppl_after_train - ppl_before:+.2f}")
print(f"  vs delta-apply delta:  {ppl_after_delta - ppl_before:+.2f}")

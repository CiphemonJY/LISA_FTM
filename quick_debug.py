#!/usr/bin/env python3
"""Quick debug: 1 client, 1 round, 20 batches — check if Option 3+4 fix works."""
import sys
sys.path.insert(0, '.')

import math
import random
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Config
MODEL_ID = "EleutherAI/pythia-70m"
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
LR = 8e-4
LORA_RANK = 4
LORA_ALPHA = 8.0
LORA_DROPOUT = 0.05
MAX_TRAIN_BATCHES = 20
MAX_TEST_BATCHES = 20
SEED = 42

# Same LoRA impl as byzantine_stress_test.py
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Module, rank: int = 4, alpha: float = 8.0, dropout: float = 0.05):
        super().__init__()
        self.weight_data = linear.weight.data.clone().float()
        self.bias_data = linear.bias.data.clone().float() if linear.bias is not None else None
        self.out_features, self.in_features = self.weight_data.shape
        self.is_conv1d = isinstance(linear, nn.Conv1d)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # Option 3 fix: both A and B initialized with small random values
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(self.out_features, rank) * 0.001)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        with torch.no_grad():
            original = nn.functional.linear(x_f32, self.weight_data, self.bias_data)
        lora_input = self.lora_dropout(x_f32)
        lora = nn.functional.linear(lora_input, self.lora_A)
        lora = nn.functional.linear(lora, self.lora_B)
        result = original + lora * self.scaling
        return result.to(orig_dtype)

    def trainable_params(self):
        return [self.lora_A, self.lora_B]


class LoraAppliedModel:
    TARGET_MODULES = [
        "query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h",
        "c_attn", "c_proj", "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "fc1", "fc2", "c_fc",
    ]

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
            for p in lora_layer.trainable_params():
                p.requires_grad = True

    def get_trainable_count(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def load_wikitext(tokenizer, max_seq=MAX_SEQ_LEN):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
        test_ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    except Exception:
        domains = ["the quick brown fox jumps over the lazy dog",
                   "machine learning neural network training data",
                   "federated learning distributed privacy client server",
                   "language model transformer attention mechanism",
                   "optimization gradient descent learning rate"]
        words = " ".join(domains).split()
        random.seed(SEED)
        train_texts = []
        for _ in range(600):
            sel = random.sample(words, min(30, len(words)))
            random.shuffle(sel)
            train_texts.append(" ".join(sel * 3)[:150])
        test_texts = []
        for _ in range(100):
            sel = random.sample(words, min(25, len(words)))
            random.shuffle(sel)
            test_texts.append(" ".join(sel * 3)[:150])
        return train_texts, test_texts
    train_texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20]
    test_texts = [t for t in test_ds["text"] if t.strip() and len(t.strip()) > 20]
    print(f"  wikitext: {len(train_texts)} train, {len(test_texts)} test lines")
    return train_texts, test_texts


def tokenize_texts(tokenizer, texts, max_seq):
    enc = tokenizer(texts, max_length=max_seq, padding="max_length", truncation=True, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": enc["input_ids"].clone()}


@torch.no_grad()
def compute_perplexity(model, test_enc, batch_size=BATCH_SIZE, max_batches=MAX_TEST_BATCHES):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    pad_token_id = getattr(model.config, "pad_token_id", None) or -100
    n_batches = min((len(test_enc["input_ids"]) + batch_size - 1) // batch_size, max_batches)
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(test_enc["input_ids"]))
        ids = test_enc["input_ids"][start:end].clone().clamp(0, model.config.vocab_size - 1)
        mask = test_enc["attention_mask"][start:end]
        labs = test_enc["labels"][start:end].clone().clamp(0, model.config.vocab_size - 1)
        outputs = model(input_ids=ids, attention_mask=mask)
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labs.view(-1))
        total_loss += loss.item() * ids.numel()
        total_tokens += ids.numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1)) if total_tokens > 0 else float("inf")


def snapshot_lora_state(wrapper):
    state = {}
    for full_name, lora_layer in wrapper.lora_layers.items():
        state[f"{full_name}.lora_A"] = lora_layer.lora_A.data.clone().cpu()
        state[f"{full_name}.lora_B"] = lora_layer.lora_B.data.clone().cpu()
    return state


def restore_lora_state(wrapper, state):
    for full_name, lora_layer in wrapper.lora_layers.items():
        lora_layer.lora_A.data.copy_(state[f"{full_name}.lora_A"].clone())
        lora_layer.lora_B.data.copy_(state[f"{full_name}.lora_B"].clone())


def compute_deltas(before, after):
    return {k: after[k] - before[k] for k in before}


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("Quick Debug: Option 3+4 fix (adaptive SERVER_LR + random B init)")
    print("=" * 60)

    print("\n[1] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config, trust_remote_code=True, torch_dtype=torch.float32)
    print(f"  Loaded {MODEL_ID}")

    print("\n[2] Loading data...")
    train_texts, test_texts = load_wikitext(tokenizer)
    test_enc = tokenize_texts(tokenizer, test_texts, MAX_SEQ_LEN)
    client_texts = train_texts[:200]  # Use first 200 texts as client data

    print("\n[3] Applying LoRA (Option 3 fix: random B init)...")
    wrapper = LoraAppliedModel(model, rank=LORA_RANK, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    wrapper.apply_lora()

    print("\n[4] ppl BEFORE training (LoRA init with random A and B)...")
    ppl_before = compute_perplexity(model, test_enc)
    print(f"  ppl = {ppl_before:.2e}")

    print("\n[5] Training: 1 client, 1 round, 20 batches...")
    wrapper.freeze_all()
    wrapper.unfreeze_all_lora()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    enc = tokenize_texts(tokenizer, client_texts, MAX_SEQ_LEN)
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    model.train()

    round_base = snapshot_lora_state(wrapper)

    n_batches = min((len(enc["input_ids"]) + BATCH_SIZE - 1) // BATCH_SIZE, MAX_TRAIN_BATCHES)
    losses = []
    for i in range(n_batches):
        idx = list(range(i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, len(enc["input_ids"]))))
        ids = enc["input_ids"][idx].clone().clamp(0, tokenizer.vocab_size - 1)
        mask = enc["attention_mask"][idx]
        labs = enc["labels"][idx].clone().clamp(0, tokenizer.vocab_size - 1)
        optimizer.zero_grad()
        outputs = model(input_ids=ids, attention_mask=mask, labels=labs)
        loss = outputs.loss
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        losses.append(loss.item())
        if i % 5 == 0:
            print(f"  batch {i}/{n_batches}: loss={loss.item():.4f}")

    state_after = snapshot_lora_state(wrapper)
    delta = compute_deltas(round_base, state_after)
    delta_norm = math.sqrt(sum(v.float().pow(2).sum().item() for v in delta.values()))
    print(f"\n  Delta norm after local training: {delta_norm:.6f}")

    print("\n[6] ppl AFTER local training (before aggregation)...")
    ppl_after_train = compute_perplexity(model, test_enc)
    print(f"  ppl = {ppl_after_train:.2e}")

    print("\n[7] Simulating aggregation with Option 4 (adaptive SERVER_LR)...")
    SERVER_LR = min(0.1, 0.01 / delta_norm) if delta_norm > 1e-8 else 0.1
    print(f"  SERVER_LR = {SERVER_LR:.6f} (delta_norm={delta_norm:.6f})")
    print(f"  Effective update scale = delta_norm * SERVER_LR = {delta_norm * SERVER_LR:.6f}")

    # Actually apply the aggregation
    for full_name, lora_layer in wrapper.lora_layers.items():
        for suffix in ["lora_A", "lora_B"]:
            key = f"{full_name}.{suffix}"
            if key in delta:
                before_norm = getattr(lora_layer, suffix).data.norm().item()
                with torch.no_grad():
                    getattr(lora_layer, suffix).add_(delta[key] * SERVER_LR)
                after_norm = getattr(lora_layer, suffix).data.norm().item()
                print(f"  {suffix} {full_name[:50]}: norm {before_norm:.6f} -> {after_norm:.6f}")

    print("\n[8] ppl AFTER aggregation (simulated server update)...")
    ppl_after_agg = compute_perplexity(model, test_enc)
    print(f"  ppl = {ppl_after_agg:.2e}")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  ppl BEFORE training:    {ppl_before:.2e}")
    print(f"  ppl AFTER local train:  {ppl_after_train:.2e}")
    print(f"  ppl AFTER aggregation:  {ppl_after_agg:.2e}")
    print(f"  Delta norm:             {delta_norm:.6f}")
    print(f"  Adaptive SERVER_LR:     {SERVER_LR:.6f}")
    print(f"  Effective step:         {delta_norm * SERVER_LR:.6f}")

    if ppl_after_agg < 1e10:
        print("\n  ✅ FIX WORKS! ppl is below 1e10")
    else:
        print(f"\n  ❌ Still diverging: ppl={ppl_after_agg:.2e}")

    # Restore state and compute final ppl properly
    restore_lora_state(wrapper, round_base)
    print(f"\n[9] ppl after RESTORING to base (should match ppl_before): {compute_perplexity(model, test_enc):.2e}")


if __name__ == "__main__":
    main()

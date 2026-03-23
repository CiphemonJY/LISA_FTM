import os, sys, time
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, os.path.dirname(__file__))

start = time.time()
MODEL_ID = "EleutherAI/pythia-70m"

print(f"Loading tokenizer for {MODEL_ID}...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"Tokenizer loaded in {time.time()-start:.1f}s")

print(f"Loading model (this downloads if needed)...")
from eval.fedavg_vs_lisafedavg import LoRALinear, LoraAppliedModel
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
print(f"Model loaded in {time.time()-start:.1f}s | Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# Apply LoRA
lora = LoraAppliedModel(model, rank=4, alpha=8.0)
count = lora.apply_lora()
print(f"LoRA applied to {count} layers")

# Quick train 1 batch per client
from eval.fedavg_vs_lisafedavg import load_wikitext, partition_data, tokenize_texts
train_texts, test_texts = load_wikitext(tok, max_seq=64)
client_texts = partition_data(train_texts, 3, non_iid=True)
print(f"Data loaded: {len(train_texts)} train texts across {len(client_texts)} clients")

# Encode one batch
batch = tokenize_texts(tok, client_texts[0][:4], max_seq=64)
print(f"Batch shape: input_ids={batch['input_ids'].shape}")
print("MODEL LOAD OK")

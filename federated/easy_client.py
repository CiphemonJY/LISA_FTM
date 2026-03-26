#!/usr/bin/env python3
"""LISA Easy Join Client"""
import sys, time, requests, argparse, pickle, base64

def join_and_train(code, server_url, client_id, steps=10):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from peft import LoraConfig, get_peft_model
    
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True, torch_dtype=torch.float32)
    lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "k_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    print(f"Model ready. Training {steps} steps...")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    texts = ["The quick brown fox jumps over the lazy dog.", "Once upon a time in a far away land.", "Machine learning is transforming the world."]
    
    for i in range(steps):
        inputs = tokenizer(texts[i % 3], return_tensors="pt", truncation=True, max_length=64)
        optimizer.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print(f"  Step {i+1}: loss={loss.item():.4f}")
    
    gradient = {n: p.grad.cpu().numpy() for n, p in model.named_parameters() if p.grad is not None}
    data = pickle.dumps(gradient)
    payload = {"client_id": client_id, "round_number": 1, "gradient_data": base64.b64encode(data).decode(), "compression_info": {"method": "none"}}
    
    print(f"Submitting gradient ({len(data):,} bytes)...")
    r = requests.post(f"{server_url}/submit", json=payload, timeout=60)
    result = r.json()
    print(f"Result: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--join", "-j")
    parser.add_argument("--server", "-s", default="http://10.0.0.43:8080")
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    client_id = args.client_id or f"jet_{int(time.time())}"
    server_url = args.server
    print(f"Joining with code {args.join} via {server_url}")
    join_and_train(args.join, server_url, client_id, args.steps)

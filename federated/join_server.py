#!/usr/bin/env python3
"""
LISA Federated Learning - Easy Join Server
BitTorrent-style share links for federated learning
"""
import os
import sys
import time
import pickle
import base64
import zlib
import hashlib
import secrets
import threading
import numpy as np
import torch
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lisa-join")

# ============ LoRA ============
def apply_lora_to_model(model, rank=4, alpha=8.0, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=dropout, bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)

# ============ Join Code System ============
JOIN_CODES = {}  # code -> {server_url, model_name, created_at}

def generate_join_code(server_url="http://localhost:8080", model_name="Qwen/Qwen2.5-0.5B"):
    """Generate a short, easy-to-type join code."""
    code = secrets.token_hex(4).upper()[:8]  # 8 chars like "A7X9K2M4"
    JOIN_CODES[code] = {
        "server_url": server_url,
        "model_name": model_name,
        "created_at": time.time(),
        "clients": 0
    }
    return code

def get_join_config(code):
    """Get server config for a join code."""
    if code in JOIN_CODES:
        JOIN_CODES[code]["clients"] += 1
        return JOIN_CODES[code]
    return None

def validate_join_code(code):
    """Check if join code exists and is valid."""
    if code in JOIN_CODES:
        age = time.time() - JOIN_CODES[code]["created_at"]
        if age < 86400:  # Valid for 24 hours
            return True
        else:
            del JOIN_CODES[code]
    return False

# ============ Federated Server ============
@dataclass
class RoundState:
    round_num: int
    status: str = "waiting"
    gradients: dict = None
    def __post_init__(self):
        self.gradients = self.gradients or {}

class FederatedServer:
    def __init__(self, model_name, checkpoint_dir="checkpoints", lr=0.01):
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.lr = lr
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.model = apply_lora_to_model(self.model)
        logger.info("Model loaded with LoRA")
        
        self.rounds: Dict[int, RoundState] = {}
        self.lock = threading.Lock()
        self.stats = {"total_gradients": 0, "total_clients": set()}
        
    def submit_gradient(self, client_id, round_number, state_dict):
        with self.lock:
            if round_number not in self.rounds:
                self.rounds[round_number] = RoundState(round_number)
            self.rounds[round_number].gradients[client_id] = state_dict
            self.rounds[round_number].status = "collecting"
            self.stats["total_gradients"] += 1
            self.stats["total_clients"].add(client_id)
        
        threading.Thread(target=self._aggregate, args=(round_number,), daemon=True).start()
        
        return {
            "status": "submitted",
            "round": round_number,
            "gradients_received": len(self.rounds[round_number].gradients)
        }
    
    def _aggregate(self, round_num):
        time.sleep(2)
        
        with self.lock:
            if round_num not in self.rounds:
                return
            state = self.rounds[round_num]
            if state.status == "aggregating":
                return
            state.status = "aggregating"
            gradients = list(state.gradients.values())
        
        logger.info(f"Aggregating round {round_num} with {len(gradients)} gradients")
        
        all_keys = set()
        for g in gradients:
            all_keys.update(g.keys())
            
        aggregated = {}
        for key in all_keys:
            grads = []
            for g in gradients:
                if key in g:
                    grad = g[key]
                    if isinstance(grad, np.ndarray):
                        grad = torch.from_numpy(grad)
                    grads.append(grad.float())
            if grads:
                aggregated[key] = torch.stack(grads).mean(0)
        
        model_state = self.model.state_dict()
        for key, grad in aggregated.items():
            if key in model_state:
                param = model_state[key]
                if isinstance(param, np.ndarray):
                    param = torch.from_numpy(param)
                model_state[key] = param.float() + self.lr * grad.float()
        
        self.model.load_state_dict(model_state)
        
        with self.lock:
            self.rounds[round_num].status = "complete"
            
        logger.info(f"Round {round_num} complete - {len(aggregated)} keys updated")
        self._save_checkpoint(round_num)
        
    def _save_checkpoint(self, round_num):
        path = os.path.join(self.checkpoint_dir, f"model_round_{round_num}.pt")
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, path)
        logger.info(f"Saved: {path}")
        
    def receive_gradient(self, data: Dict) -> Dict:
        client_id = data.get("client_id", "unknown")
        round_num = data.get("round_number", 1)
        gradient_b64 = data.get("gradient_data", "")
        compression_info = data.get("compression_info", {})
        
        try:
            gradient_bytes = base64.b64decode(gradient_b64)
            method = compression_info.get("method", "none")
            if method != "none":
                try:
                    gradient_bytes = zlib.decompress(gradient_bytes)
                except:
                    pass
            state_dict = pickle.loads(gradient_bytes)
        except Exception as e:
            return {"status": "error", "message": str(e)}
            
        return self.submit_gradient(client_id, round_num, state_dict)
    
    def get_status(self):
        with self.lock:
            rounds = {rn: {"status": rs.status, "gradients": len(rs.gradients)} 
                     for rn, rs in self.rounds.items()}
            return {
                "rounds": rounds,
                "total_gradients": self.stats["total_gradients"],
                "total_clients": len(self.stats["total_clients"])
            }

# ============ Web App ============
app = FastAPI()
server = None

@app.get("/")
async def root():
    return {"status": "ok", "server": "lisa-join", "version": "1.0"}

@app.get("/join/{code}")
async def join_page(code: str):
    """Web page to join the federated network."""
    config = get_join_config(code.upper())
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LISA Federated Learning - Join</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                   max-width: 600px; margin: 50px auto; padding: 20px; text-align: center;
                   background: #1a1a2e; color: #eee; }}
            .card {{ background: #16213e; border-radius: 16px; padding: 40px; box-shadow: 0 8px 32px rgba(0,0,0,0.3); }}
            h1 {{ color: #00d9ff; margin-bottom: 10px; }}
            .subtitle {{ color: #888; margin-bottom: 30px; }}
            .code {{ background: #0f3460; padding: 15px 30px; border-radius: 8px;
                    font-family: monospace; font-size: 24px; letter-spacing: 4px;
                    color: #00d9ff; margin: 20px 0; }}
            .cmd {{ background: #000; padding: 15px; border-radius: 8px;
                   font-family: monospace; font-size: 14px; text-align: left; overflow-x: auto;
                   color: #0f0; margin: 20px 0; }}
            .ok {{ color: #00ff88; }}
            .info {{ color: #888; font-size: 14px; margin-top: 20px; }}
            a {{ color: #00d9ff; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🤝 LISA Federated Learning</h1>
            <p class="subtitle">Join the distributed AI training network</p>
            
            <div class="code">{code.upper()}</div>
            
            <p>Your join code is ready! Run this on any device:</p>
            
            <div class="cmd">
                curl -sL https://lisa.ciphemon.ai/install | bash -s {code.upper()}
            </div>
            
            <p class="info">
                Or with Python directly:<br>
                <code>pip install lisa-client && lisa-client --join {code.upper()}</code>
            </p>
            
            <hr style="border-color: #333; margin: 30px 0;">
            
            <h3>How it works:</h3>
            <p style="text-align: left; color: #aaa;">
                1. Your device downloads the LISA client<br>
                2. Client connects to server using your code<br>
                3. Client trains on local data (data never leaves your device)<br>
                4. Only gradient updates are shared<br>
                5. All participants benefit from collective training
            </p>
            
            <p class="info" style="margin-top: 30px;">
                💡 Your data stays private - only model gradients are shared
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/api/config/{code}")
async def get_config(code: str):
    """API endpoint for client to get server config."""
    config = get_join_config(code.upper())
    if config:
        return {
            "status": "ok",
            "server_url": config["server_url"],
            "model_name": config["model_name"]
        }
    raise HTTPException(status_code=404, detail="Invalid or expired join code")

@app.get("/health")
async def health():
    return {"status": "ok", "server": "lisa-join"}

@app.post("/submit")
async def submit(data: dict):
    return server.receive_gradient(data)

@app.get("/status")
async def status():
    return server.get_status()

# ============ Main ============
def main():
    import argparse
    parser = argparse.ArgumentParser(description="LISA Federated Learning Server with Easy Join")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--generate-code", action="store_true", help="Generate a join code")
    args = parser.parse_args()
    
    global server
    server = FederatedServer(args.model, args.checkpoint_dir, args.lr)
    
    # Generate join code if requested
    if args.generate_code:
        # Try to determine public URL
        public_url = f"http://localhost:{args.port}"
        code = generate_join_code(public_url, args.model)
        print(f"\n{'='*50}")
        print(f"🎉 JOIN CODE READY!")
        print(f"{'='*50}")
        print(f"\n  Code: {code}")
        print(f"  URL:  http://YOUR_IP:{args.port}/join/{code}")
        print(f"\n  Share this code to let others join!")
        print(f"\n  Or visit: http://localhost:{args.port}/join/{code}")
        print(f"{'='*50}\n")
    
    logger.info(f"Starting LISA server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()

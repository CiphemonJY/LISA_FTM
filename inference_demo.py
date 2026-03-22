#!/usr/bin/env python3
"""
Clean inference demo for LISA_FTM project.

Loads a checkpoint saved by real_training.py (Pythia-70m + LoRA) or
offload_test.py (TinyLlama + LoRA) and runs text generation.

Key fix: always use the config embedded WITHIN the checkpoint directory
(model.safetensors + config.json), or fall back to the original model ID
with the correct hidden_size. Never override hidden_size with a mismatched
value.
"""
import os
import sys
import logging
from pathlib import Path

import torch

# Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("inference-demo")

DEVICE = "cpu"
DTYPE = torch.float32


def find_latest_checkpoint(output_dir: Path):
    """Find the latest .pt checkpoint in output dir."""
    checkpoints = list(output_dir.glob("*.pt"))
    if not checkpoints:
        return None
    # Prefer final_model.pt, else latest step
    final = output_dir / "final_model.pt"
    if final.exists():
        return final
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def load_pythia_lora_model(checkpoint_dir: Path, checkpoint_name: str = None):
    """
    Load a Pythia-70m model with LoRA from a checkpoint directory.

    Supports two directory layouts:
    1. Full (safetensors + config): model.safetensors, config.json, tokenizer files
    2. Training output (pt only): *.pt checkpoint, no config.json

    For layout 2, we load the base model from HuggingFace and overlay the .pt state.
    The checkpoint keys use "gpt_neox.xxx" format matching HuggingFace EleutherAI/pythia-70m.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config_path = checkpoint_dir / "config.json"
    safetensors_path = checkpoint_dir / "model.safetensors"

    # Determine base model source
    base_model_id = "EleutherAI/pythia-70m"
    has_full_structure = config_path.exists() and safetensors_path.exists()

    if has_full_structure:
        # Layout 1: load config from local directory (correct hidden_size is here)
        log.info(f"Loading config from {config_path}")
        config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
        base_source = str(checkpoint_dir)
    else:
        # Layout 2: no config.json/safetensors — use HuggingFace base
        # Inspect checkpoint FIRST to get exact shapes (vocab may differ after resize)
        ckpt_candidate = checkpoint_dir / (checkpoint_name or "final_model.pt")
        if not ckpt_candidate.exists():
            ckpt_candidate = find_latest_checkpoint(checkpoint_dir)
        if ckpt_candidate and ckpt_candidate.exists():
            log.info(f"  Inspecting checkpoint: {ckpt_candidate.name}")
            ckpt = torch.load(ckpt_candidate, map_location="cpu", weights_only=True)
            # Extract vocab_size from embedding weight shape
            ckpt_vocab_size = None
            ckpt_hidden_size = None
            for k, v in list(ckpt.items())[:10]:
                if isinstance(v, torch.Tensor) and v.dim() == 2:
                    if "embed" in k or "lm_head" in k:
                        ckpt_vocab_size = v.shape[0]
                        ckpt_hidden_size = v.shape[1]
                        log.info(f"  Found weight matrix {k}: {v.shape} -> vocab={ckpt_vocab_size}, hidden={ckpt_hidden_size}")
                        break
            del ckpt
        else:
            ckpt_vocab_size = None
            ckpt_hidden_size = None

        log.info(f"No local config found, loading base model from {base_model_id}")
        base_config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)

        # Override with exact checkpoint shapes (model was resized during training)
        if ckpt_vocab_size is not None:
            log.info(f"  Overriding vocab_size: {base_config.vocab_size} -> {ckpt_vocab_size}")
            base_config.vocab_size = ckpt_vocab_size
        if ckpt_hidden_size is not None:
            log.info(f"  Overriding hidden_size: {base_config.hidden_size} -> {ckpt_hidden_size}")
            base_config.hidden_size = ckpt_hidden_size

        config = base_config
        base_source = base_model_id

    log.info(f"  Model type: {config.model_type}")
    log.info(f"  Hidden size: {config.hidden_size}")
    log.info(f"  Num layers: {config.num_hidden_layers}")
    log.info(f"  Vocab size: {config.vocab_size}")

    # Load base model
    log.info(f"Loading base model from {base_source}")
    model = AutoModelForCausalLM.from_pretrained(
        base_source,
        config=config,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        ignore_mismatched_sizes=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Base model loaded: {n_params:,} params")

    # Load tokenizer
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if tokenizer_path.exists():
        log.info(f"Loading tokenizer from {checkpoint_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint_dir), trust_remote_code=True
        )
    else:
        log.info(f"Loading tokenizer from {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoint state dict if present
    if checkpoint_name:
        ckpt_path = checkpoint_dir / checkpoint_name
    else:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)

    if ckpt_path and ckpt_path.exists():
        log.info(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        log.info(f"  Checkpoint keys: {len(state)}")

        # Keys use "gpt_neox.xxx" prefix matching HuggingFace format
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning(f"  Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Missing keys: {missing}")
        if unexpected:
            log.warning(f"  Unexpected keys (LoRA/embedding resizing is normal): {unexpected[:5]}...")

        log.info(f"  Checkpoint loaded successfully")
    else:
        log.info(f"  No checkpoint found, using base model")

    return model, tokenizer, config


def load_tinyllama_lora_model(checkpoint_dir: Path, checkpoint_name: str = None):
    """
    Load a TinyLlama model with LoRA from a checkpoint directory.

    TinyLlama uses the Llama architecture. Supports same two layouts as Pythia.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config_path = checkpoint_dir / "config.json"
    safetensors_path = checkpoint_dir / "model.safetensors"

    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    has_full_structure = config_path.exists() and safetensors_path.exists()

    if has_full_structure:
        log.info(f"Loading config from {config_path}")
        config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
        base_source = str(checkpoint_dir)
    else:
        log.info(f"No local config found, loading base model from {base_model_id}")
        config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
        base_source = base_model_id

    log.info(f"  Model type: {config.model_type}")
    log.info(f"  Hidden size: {config.hidden_size}")
    log.info(f"  Num layers: {config.num_hidden_layers}")

    # Load base model
    log.info(f"Loading base model from {base_source}")
    model = AutoModelForCausalLM.from_pretrained(
        base_source,
        config=config,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        ignore_mismatched_sizes=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Base model loaded: {n_params:,} params")

    # Load tokenizer
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoint
    if checkpoint_name:
        ckpt_path = checkpoint_dir / checkpoint_name
    else:
        ckpt_path = find_latest_checkpoint(checkpoint_dir)

    if ckpt_path and ckpt_path.exists():
        log.info(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        log.info(f"  Checkpoint keys: {len(state)}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning(f"  Missing keys: {missing[:5]}")
        if unexpected:
            log.warning(f"  Unexpected keys: {unexpected[:5]}")
        log.info(f"  Checkpoint loaded successfully")
    else:
        log.info(f"  No checkpoint found, using base model")

    return model, tokenizer, config


def run_inference(model, tokenizer, prompts, max_new_tokens: int = 30, temperature: float = 0.8):
    """Run text generation on a list of prompts."""
    model.eval()

    results = []
    for prompt in prompts:
        log.info(f"\n  Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        # Clamp to valid vocab range
        inputs = {k: v.clamp(0, tokenizer.vocab_size - 1) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        generated = outputs[0]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        # Sanitize for Windows console
        safe = text.encode("cp1252", errors="replace").decode("cp1252")
        log.info(f"  Output: {safe}")
        results.append({"prompt": prompt, "output": safe})

    return results


def detect_model_type(checkpoint_dir: Path):
    """Detect whether checkpoint is Pythia (GPT-NeoX) or TinyLlama (Llama)."""
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        if "gpt2" in model_type.lower() or "gpt-neox" in model_type.lower():
            return "pythia"
        elif "llama" in model_type.lower():
            return "tinyllama"
        # Check hidden_size as fallback clue
        n_embd = cfg.get("n_embd", cfg.get("hidden_size", 0))
        if n_embd == 512:
            return "pythia"
        elif n_embd == 2048:
            return "tinyllama"
    # No config — inspect checkpoint key format
    ckpt = find_latest_checkpoint(checkpoint_dir)
    if ckpt and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        first_key = next(iter(state.keys()), "")
        if "gpt_neox" in first_key:
            return "pythia"
        elif "model.embed_tokens" in first_key or "lm_head" in first_key:
            return "tinyllama"
    return "unknown"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LISA_FTM Inference Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pt file (or directory containing checkpoint)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="output/real_training",
        help="Checkpoint directory (default: output/real_training)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "PyTorch is a",
            "Machine learning models",
            "The history of artificial",
        ],
        help="Prompts for generation",
    )
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LISA_FTM INFERENCE DEMO")
    log.info("=" * 60)

    # Resolve checkpoint directory
    root = Path(__file__).parent
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.is_dir():
            checkpoint_dir = ckpt_path
        else:
            checkpoint_dir = ckpt_path.parent
            ckpt_name = ckpt_path.name
    else:
        checkpoint_dir = root / args.dir
        ckpt_name = None

    log.info(f"Checkpoint dir: {checkpoint_dir}")

    if not checkpoint_dir.exists():
        log.error(f"Checkpoint directory not found: {checkpoint_dir}")
        log.info("Available checkpoints:")
        for d in sorted((root / "output").iterdir()):
            if d.is_dir():
                final = d / "final_model.pt"
                steps = list(d.glob("step_*.pt"))
                log.info(f"  {d.name}: final={'yes' if final.exists() else 'no'}, steps={[s.name for s in steps]}")
        return

    # Detect model type
    model_type = detect_model_type(checkpoint_dir)
    log.info(f"Detected model type: {model_type}")

    # Load model
    try:
        if model_type == "pythia":
            model, tokenizer, config = load_pythia_lora_model(checkpoint_dir, ckpt_name)
        elif model_type == "tinyllama":
            model, tokenizer, config = load_tinyllama_lora_model(checkpoint_dir, ckpt_name)
        else:
            # Try Pythia first (more likely), fall back to TinyLlama
            try:
                model, tokenizer, config = load_pythia_lora_model(checkpoint_dir, ckpt_name)
            except Exception as e1:
                log.warning(f"Pythia load failed: {e1}")
                try:
                    model, tokenizer, config = load_tinyllama_lora_model(checkpoint_dir, ckpt_name)
                except Exception as e2:
                    log.error(f"TinyLlama load also failed: {e2}")
                    raise
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"\nModel ready: {n_params:,} params, device={DEVICE}")

    # Run inference
    log.info(f"\nGenerating text ({args.max_tokens} tokens, temp={args.temperature})...")
    results = run_inference(
        model, tokenizer, args.prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for r in results:
        log.info(f"  Prompt: {r['prompt']}")
        log.info(f"  Output: {r['output']}\n")


if __name__ == "__main__":
    main()

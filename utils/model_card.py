"""
Model card generator for LISA_FTM federated training runs.
Generates MODEL_CARD.md and model_card.json from checkpoint metadata.
"""
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def generate_model_card(
    checkpoint_path: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a model card from checkpoint metadata.

    Args:
        checkpoint_path: Path to checkpoint directory (e.g. "checkpoints/round_5_v1")
        output_path: Optional path for the .md output.
                     Defaults to <checkpoint_path>/MODEL_CARD.md.

    Returns:
        Path to the generated MODEL_CARD.md file.
    """
    ckpt_path = Path(checkpoint_path)
    if output_path:
        out_path = Path(output_path)
    else:
        out_path = ckpt_path / "MODEL_CARD.md"

    meta = _load_metadata(ckpt_path)
    config = _load_config(ckpt_path)

    card_data = _build_card_data(meta, config, ckpt_path)
    markdown = _render_markdown(card_data, meta)

    # Save markdown
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    # Save JSON sidecar
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(card_data, f, indent=2, default=str)

    return str(out_path)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _load_metadata(ckpt_path: Path) -> dict:
    meta_path = ckpt_path / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_config(ckpt_path: Path) -> dict:
    # Try config.json alongside metadata, then fall back to config/default.yaml
    config_path = ckpt_path / "config.json"
    if not config_path.exists():
        root = ckpt_path.parents[1]  # checkpoints/ -> project root
        config_path = root / "config" / "default.yaml"
    if config_path.suffix == ".yaml":
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    elif config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_layer_selection_stats(ckpt_path: Path) -> Optional[dict]:
    # Check for layer selection artifacts alongside the checkpoint
    for name in ("layer_selection.json", "layer_stats.json", "selected_layers.json"):
        p = ckpt_path / name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


def _build_card_data(meta: dict, config: dict, ckpt_path: Path) -> dict:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    fed_cfg = config.get("federated", {})
    privacy_cfg = config.get("privacy", {})

    layer_stats = _load_layer_selection_stats(ckpt_path)

    # Derive trainable param percentage from metadata if available
    trainable_pct = None
    extra = meta.get("extra", {})
    if isinstance(extra, dict):
        trainable_pct = extra.get("trainable_params_pct")

    # DP settings
    dp_enabled = meta.get("dp_enabled", privacy_cfg.get("enabled", False))
    dp_epsilon = extra.get("dp_epsilon") if isinstance(extra, dict) else None
    if dp_epsilon is None and dp_enabled:
        dp_epsilon = privacy_cfg.get("epsilon")

    # Compression / byzantine
    compression = meta.get("compression", "none")
    byzantine_method = None
    if isinstance(extra, dict):
        byzantine_method = extra.get("byzantine_method")

    # Timestamps
    ts = meta.get("timestamp_iso") or meta.get("timestamp")
    if ts:
        try:
            if isinstance(ts, (int, float)):
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            generated_at = dt.isoformat()
        except Exception:
            generated_at = str(ts)
    else:
        generated_at = datetime.now(timezone.utc).isoformat()

    card = {
        "model_details": {
            "architecture": "LoRA-finetuned transformer",
            "base_model": model_cfg.get("name", "EleutherAI/pythia-70m"),
            "lora_rank": model_cfg.get("lora_rank"),
            "lora_alpha": model_cfg.get("lora_alpha"),
            "trainable_params_pct": trainable_pct,
        },
        "training": {
            "federated_rounds": meta.get("round"),
            "local_epochs": training_cfg.get("epochs"),
            "local_steps": fed_cfg.get("local_steps"),
            "learning_rate": training_cfg.get("lr"),
            "batch_size": training_cfg.get("batch_size"),
            "seq_length": training_cfg.get("seq_length"),
            "clients_per_round": meta.get("client_count"),
        },
        "privacy": {
            "differential_privacy_enabled": dp_enabled,
            "dp_epsilon": dp_epsilon,
            "compression": compression,
            "byzantine_method": byzantine_method,
        },
        "performance": {
            "perplexity": meta.get("perplexity"),
            "avg_gradient_norm": meta.get("avg_gradient_norm"),
            "round_time_seconds": meta.get("round_time"),
            "checkpoint_id": meta.get("checkpoint_id"),
        },
        "layer_selection": layer_stats,
    }

    return card


def _render_markdown(card: dict, meta: dict) -> str:
    lines = []
    W = 90

    def section(title: str) -> None:
        lines.append(f"\n## {title}\n")

    def kv(key: str, val: Any) -> None:
        if val is None:
            return
        lines.append(f"- **{key}:** {val}")

    # Header
    ckpt_id = card["performance"].get("checkpoint_id") or "unknown"
    lines.append(f"# Model Card — {ckpt_id}\n")
    lines.append(f"_Auto-generated by LISA_FTM — {datetime.now(timezone.utc).isoformat()}_\n")

    # Model Details
    section("Model Details")
    md = card["model_details"]
    kv("Base model", md["base_model"])
    kv("Architecture", md["architecture"])
    kv("LoRA rank (r)", md["lora_rank"])
    kv("LoRA alpha", md["lora_alpha"])
    tp = md["trainable_params_pct"]
    if tp is not None:
        kv("Trainable parameters", f"{tp:.2%}")
    kv("Checkpoint ID", md.get("architecture") or ckpt_id)

    # Training
    section("Training Details")
    t = card["training"]
    kv("Federated rounds completed", t["federated_rounds"])
    kv("Local epochs per client", t["local_epochs"])
    kv("Local steps per round", t["local_steps"])
    kv("Learning rate", t["learning_rate"])
    kv("Batch size", t["batch_size"])
    kv("Sequence length", t["seq_length"])
    kv("Clients per round", t["clients_per_round"])

    # Privacy
    section("Privacy & Robustness")
    p = card["privacy"]
    kv("Differential privacy enabled", p["differential_privacy_enabled"])
    if p["dp_epsilon"] is not None:
        kv("DP epsilon (ε)", p["dp_epsilon"])
    kv("Gradient compression", p["compression"])
    kv("Byzantine-resilience method", p["byzantine_method"])

    # Performance
    section("Performance")
    perf = card["performance"]
    ppl = perf.get("perplexity")
    if ppl is not None:
        kv("Perplexity (validation)", f"{ppl:.4f}")
    kv("Avg. gradient norm", perf.get("avg_gradient_norm"))
    kv("Round time (s)", perf.get("round_time_seconds"))
    kv("Checkpoint ID", perf.get("checkpoint_id"))

    # Layer Selection
    ls = card["layer_selection"]
    if ls:
        section("Layer Selection (LISA)")
        for k, v in ls.items():
            if isinstance(v, (list, dict)):
                lines.append(f"- **{k}:** {json.dumps(v)}")
            else:
                lines.append(f"- **{k}:** {v}")

    # How to Use
    section("How to Use")
    lines.append(textwrap.fill(
        "Load this checkpoint using `CheckpointManager` from `utils.checkpoint_manager`:",
        width=W,
    ))
    lines.append("\n```python")
    lines.append("from utils.checkpoint_manager import CheckpointManager")
    lines.append(f('checkpoint_id = "{ckpt_id}"').format(ckpt_id=ckpt_id))
    lines.append("cm = CheckpointManager()")
    lines.append("data = cm.load(checkpoint_id)")
    lines.append("model.load_state_dict(data['model'])")
    lines.append("```")
    lines.append("\nOr load the weights directly with `torch.load`:")
    lines.append("\n```python")
    lines.append(f('state = torch.load("{ckpt_id}/model.pt", map_location="cpu")')
                  .format(ckpt_id=ckpt_id))
    lines.append("```")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Public entry-point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a model card from a checkpoint.")
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Checkpoint directory (e.g. checkpoints/round_5_v1)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output .md path (defaults to <checkpoint>/MODEL_CARD.md)",
    )
    args = parser.parse_args()

    out = generate_model_card(args.checkpoint, args.output)
    print(f"Model card written to: {out}")

#!/usr/bin/env python3
"""
CLI wrapper for generating a model card from a LISA_FTM checkpoint.

Usage:
    python scripts/generate_model_card.py --checkpoint checkpoints/round_5_v1
    python scripts/generate_model_card.py -c checkpoints/round_5_v1 -o custom_path.md
"""
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.model_card import generate_model_card


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a LISA_FTM model card from a checkpoint directory."
    )
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

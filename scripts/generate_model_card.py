#!/usr/bin/env python3
"""
Standalone CLI for generating a LISA_FTM Model Card.

Generates MODEL_CARD.md and model_card.json from a federated training checkpoint,
reading experiment results, config, and checkpoint metadata.

Usage:
    # Use the latest checkpoint
    python scripts/generate_model_card.py

    # Specify a checkpoint
    python scripts/generate_model_card.py --checkpoint checkpoints/round_5_v1

    # Override output path
    python scripts/generate_model_card.py --checkpoint checkpoints/round_5_v1 \\
        --output checkpoints/round_5_v1/MODEL_CARD.md

    # Multiple config files (first found is used)
    python scripts/generate_model_card.py --config config/default.yaml config/prod.yaml

Exit codes:
    0  - success
    1  - checkpoint not found
    2  - generation failed
"""

import sys
from pathlib import Path

# Add project root to path so `utils.model_card` resolves
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_card import generate


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a LISA_FTM model card from a federated training checkpoint.",
        epilog=(
            "If --checkpoint points to a directory that doesn't exist, the script "
            "automatically uses the most recent checkpoint under checkpoints/. "
            "Missing data is handled gracefully — unavailable sections are omitted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="checkpoints",
        metavar="PATH",
        help=(
            "Path to the checkpoint directory (default: checkpoints). "
            "Use 'checkpoints' (no trailing slash) to automatically pick the latest. "
            "Examples: checkpoints/round_5_v1, /absolute/path/to/checkpoint"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Output path for MODEL_CARD.md (default: <checkpoint>/MODEL_CARD.md). "
            "The model_card.json is always written to the checkpoint directory."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=None,
        metavar="PATH",
        help="Config file path(s) in order of priority (default: config/default.yaml)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        metavar="PATH",
        help="Project root directory (default: auto-detected from script location)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only write model_card.json, skip MODEL_CARD.md",
    )
    parser.add_argument(
        "--md-only",
        action="store_true",
        help="Only write MODEL_CARD.md, skip model_card.json",
    )

    args = parser.parse_args()
    argv_checkpoint = args.checkpoint
    argv_output = args.output
    argv_config = args.config
    argv_project_root = args.project_root
    json_only = args.json_only
    md_only = args.md_only

    # Resolve project root
    project_root = Path(argv_project_root) if argv_project_root else PROJECT_ROOT

    # Resolve config paths
    config_paths = None
    if argv_config:
        config_paths = [Path(p) for p in argv_config]

    # -------------------------------------------------------------------------
    # Generate the model card(s)
    # -------------------------------------------------------------------------
    try:
        result = generate(
            checkpoint_path=argv_checkpoint,
            config_paths=config_paths,
            project_root=project_root,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error during generation: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

    # -------------------------------------------------------------------------
    # Handle output redirection
    # -------------------------------------------------------------------------
    md_path = result["markdown"]
    json_path = result["json"]

    if md_only:
        # Remove the markdown output if user asked for md-only
        md_path.unlink(missing_ok=True)
        print(f"model_card.json written to: {json_path}")

    elif argv_output:
        # Copy markdown to user-specified location
        import shutil
        dest = Path(argv_output)
        shutil.copy2(md_path, dest)
        print(f"MODEL_CARD.md written to: {dest}")

    else:
        print(f"MODEL_CARD.md written to: {md_path}")
        print(f"model_card.json written to: {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

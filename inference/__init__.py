"""Inference Module - LISA_FTM Model Inference

This module provides checkpoint loading and text generation for LISA_FTM
training outputs (Pythia-70m + Lo, TinyLlama + LoRA).

The main entry point is:
    from inference.engine import load_checkpoint, run_generation
"""

from .engine import (
    load_checkpoint,
    detect_model_type,
    run_generation,
    generate,
    find_latest_checkpoint,
    inspect_checkpoint,
)

# Backward-compat: re-export the few surviving simulator classes
# (the old LISAInference/KVCache/InferenceConfig were a non-functional simulator)
from .engine import LISAInference, InferenceConfig, KVCache

__all__ = [
    "load_checkpoint",
    "detect_model_type",
    "run_generation",
    "generate",
    "find_latest_checkpoint",
    "inspect_checkpoint",
    # Backward compat (no-ops / stubs from old simulator)
    "LISAInference",
    "InferenceConfig",
    "KVCache",
]

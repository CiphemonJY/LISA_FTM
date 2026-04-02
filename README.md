# LISA 120B Scale - Jetson Implementation

Training large language models on limited RAM using LISA, MoE, and LoRA techniques.

## Quick Start

```bash
# Connect to Jetson
ssh jetson@YOUR_JETSON_IP

# Run the training script
python3 /tmp/lisa_moe_improve/train_final.py
```

## Architecture

### LISA (Layer-wise Importance Sampling)
- Train only 2 of N layers simultaneously
- Reduces memory by ~N/2×
- Implemented in `LISA` class

### MoE (Mixture of Experts)
- 8 experts with top-2 routing
- Only activates subset per token
- Memory efficient for large models

### LoRA (Low-Rank Adaptation)
- rank=2-4, alpha=4-8
- Only trains adapter weights
- Base model stays frozen

## Results

| Metric | Value |
|--------|-------|
| Cross-Entropy Loss | 4.49 → 3.98 (-11%) |
| Perplexity | 89.0 → 53.6 (-40%) |
| Adapter Size | 0.64MB |
| Memory Usage | ~1.6GB |

## Files

```
jetson_scripts/
├── train_final.py     # Production training with CE loss
├── lisa_v7.py         # Latest implementation
└── ...

jetson_weights/
# Real extracted weights from Qwen2.5-14B GGUF
```

## Hardware

- **Jetson Orin**: 7.4GB RAM, 8GB GPU
- **GPU Status**: Broken (needs reboot)
- **NVMe**: 200GB available

## Limitations

- True 120B training requires multi-GPU
- Single 7.4GB cannot hold full 120B fp16
- Need actual MoE GGUF file for 120B-scale

## Research Notes

See `PROGRESS_REPORT_2026-04-01.md` for full details.

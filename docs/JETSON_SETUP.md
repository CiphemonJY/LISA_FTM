# Jetson Setup Guide

This guide covers setting up LISA training on NVIDIA Jetson Orin (8GB GPU, 7GB RAM).

## Prerequisites

- NVIDIA Jetson Orin with Ubuntu 22.04
- Python 3.8+
- 23GB swap space (required for 7B+ models)

## 1. System Setup

### 1.1 Configure Swap

```bash
# Check current swap
free -h

# Create 23GB swap file if needed
sudo fallocate -l 23G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Add to fstab for persistence
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 1.2 Install Python Dependencies

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install Python if not present
python3 --version  # Should be 3.8+

# Install pip
sudo apt install -y python3-pip

# Install PyTorch (CPU version for Jetson)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install transformers and dependencies
pip3 install transformers peft huggingface_hub accelerate
```

### 1.3 Verify Installation

```bash
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

## 2. Clone the Repository

```bash
git clone https://github.com/CiphemonJY/LISA_FTM.git
cd LISA_FTM
```

## 3. Training Quick Start

### 7B Model (Qwen2.5-7B)

```bash
# Set environment variables
export JETSON_IP="10.0.0.145"  # Your Jetson's IP
export JETSON_USER="jetson"      # Your Jetson username

# Run training
python3 lisa_lcsb/train_7b_lisa_lcsb.py --steps 500
```

### 14B Model (Qwen2.5-14B)

Requires 23GB swap space.

```bash
# Create swap if not exists
sudo fallocate -l 23G /swapfile
sudo swapon /swapfile

# Run 14B training
python3 lisa_lcsb/14b_lisa_lcsb_fullcpu.py --steps 500
```

**Note:** 14B training on Jetson takes ~10-12 hours for 500 steps.

## 4. Understanding the Output

```
[2026-03-29 13:29:15] ============================================================
[2026-03-29 13:29:15] 14B LISA+LCSB - FULL CPU MODE
[2026-03-29 13:29:15] ============================================================
[2026-03-29 13:29:15] Model: Qwen/Qwen2.5-14B
[2026-03-29 13:29:15] 
[2026-03-29 13:29:15] [1] Loading model on CPU (no offload)...
[2026-03-29 13:29:15]     This uses ~28GB RAM, will swap heavily
...
[2026-03-29 13:31:16] Step 1/500: layer=47, loss=3.1916, fwd=19.6s, bwd=100.9s
[2026-03-29 13:32:25] Step 2/500: layer=46, loss=4.3283, fwd=17.5s, bwd=50.9s
```

- **layer=46/47**: LISA cycling between last 2 layers
- **loss**: Decreasing = learning
- **fwd**: Forward pass time (seconds)
- **bwd**: Backward pass time (seconds)

## 5. Monitoring Training

### Check Progress

```bash
# Watch training log
tail -f /tmp/lisa_14b_fullcpu_stdout.txt

# Check memory usage
free -h

# Check GPU (if working)
nvidia-smi
```

### Checkpoints

Checkpoints are saved every 50 steps:
```bash
ls -la /tmp/lisa_14b_fullcpu_checkpoints/
```

## 6. Troubleshooting

### Problem: CUDA out of memory

Jetson Orin has a CUDA allocator bug. **Solution:** Use CPU-only training:
```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # Use CPU, not GPU
    trust_remote_code=True,
)
```

### Problem: Model too large

If model loading fails, try:
1. Increase swap: `sudo fallocate -l 30G /swapfile && sudo swapon /swapfile`
2. Use a smaller model (Qwen2.5-3B instead of 7B/14B)
3. Close other applications to free RAM

### Problem: Training is slow

- 7B model: ~58 seconds/step
- 14B model: ~80 seconds/step
- This is normal for CPU-only training on Jetson

Expected training times:
| Model | Steps | Time |
|-------|-------|------|
| 7B | 500 | ~8 hours |
| 14B | 500 | ~11 hours |

### Problem: Loss is NaN

This usually means sequence length is too short. Ensure sequences have 5+ tokens.

## 7. Hardware Specifications

### Jetson Orin (8GB)

- **GPU**: NVIDIA Orin, 8GB GDDR6
- **CPU**: 8-core ARM Cortex-A78AE
- **RAM**: 8GB (7.4Gi usable)
- **Storage**: NVMe SSD recommended
- **Swap**: 23GB recommended for large models

### Memory Requirements

| Model | Precision | RAM Needed | Swap Needed |
|-------|-----------|------------|-------------|
| 7B | bfloat16 | ~4GB | ~10GB |
| 14B | bfloat16 | ~8GB | ~20GB |

## 8. Next Steps

After training completes:

1. **Evaluate the model**: See `docs/EVALUATION.md`
2. **Run inference**: See `docs/INFERENCE.md`
3. **Join federated network**: See `docs/FEDERATED.md`

## Support

For issues, check:
1. GitHub Issues: https://github.com/CiphemonJY/LISA_FTM/issues
2. Training log: `/tmp/lisa_*_stdout.txt`

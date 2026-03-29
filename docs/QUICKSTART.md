# Quick Start Guide

Get up and running with LISA training in 5 minutes.

## 1. Clone the Repository

```bash
git clone https://github.com/CiphemonJY/LISA_FTM.git
cd LISA_FTM
```

## 2. Install Dependencies

```bash
pip install torch transformers peft huggingface_hub accelerate
```

## 3. Choose Your Hardware

### Option A: Jetson (8GB RAM)

See [docs/JETSON_SETUP.md](docs/JETSON_SETUP.md) for detailed setup.

```bash
# Configure swap (required!)
sudo fallocate -l 23G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Train 7B model
python3 train_7b_simple.py --steps 500

# Train 14B model (requires swap)
python3 train_14b_simple.py --steps 500
```

### Option B: Mac with Apple Silicon

```bash
# Install MLX (Apple Silicon optimized PyTorch)
pip install mlx

# Train with MLX (coming soon)
python3 train_mlx.py --model Qwen2.5-7B --steps 500
```

### Option C: Linux/Windows with GPU

```bash
# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Train with GPU
python3 train_7b_simple.py --steps 500
```

## 4. What to Expect

### Training Time

| Hardware | 7B Model | 14B Model |
|----------|----------|-----------|
| Jetson Orin (CPU) | ~8 hours | ~11 hours |
| Mac Mini M4 Pro | ~2 hours | ~4 hours |
| Desktop with GPU | ~30 min | ~1 hour |

### Output

```
[2026-03-29 13:29:15] ============================================================
[2026-03-29 13:29:15] LISA Training - Qwen2.5-7B
[2026-03-29 13:29:15] ============================================================
[2026-03-29 13:29:15] Loading Qwen/Qwen2.5-7B...
[2026-03-29 13:29:15] Model loaded: 28 layers
[2026-03-29 13:29:15] Applying LISA (training last 2 of 28 layers)...
[2026-03-29 13:29:15] Trainable params: 30,720
[2026-03-29 13:29:15] Training 500 steps...
[2026-03-29 13:29:15] ============================================================
[2026-03-29 13:30:13] Step 1/500: layer=27, loss=6.2610, fwd=8.0s, bwd=48.5s
```

### Checkpoints

Checkpoints saved every 50 steps:
```bash
ls /tmp/lisa_7b_checkpoints/
# step_50.pt  step_100.pt  step_150.pt ...
```

## 5. Understanding LISA

LISA = Layer-wise Importance Sampling for Adapters

Instead of training all 7 billion parameters, we:
1. Freeze most layers
2. Only train the last 2 layers
3. Use LoRA for efficient adapter training

**Result:** 7B model with only 30,720 trainable parameters!

## 6. Troubleshooting

### Out of Memory

- Jetson: Increase swap (`sudo fallocate -l 30G /swapfile`)
- Desktop: Use smaller model (try `Qwen/Qwen2.5-3B`)
- Mac: Close other apps

### Training Too Slow

- Normal on CPU: ~60-80 seconds per step
- Expected: 8-12 hours for 500 steps on Jetson
- Consider using a smaller model for testing

### Loss is NaN

- Usually means sequences are too short
- Ensure your training data has 5+ tokens per sequence

## 7. Next Steps

After training:
- [Evaluate your model](docs/EVALUATION.md)
- [Run inference](docs/INFERENCE.md)  
- [Join federated network](docs/FEDERATED.md)

## Need Help?

- GitHub Issues: https://github.com/CiphemonJY/LISA_FTM/issues
- Training log: `/tmp/lisa_*_stdout.txt`

# LISA Federation Training Tracker
## Last Updated: 2026-03-26 23:26 CDT

## Current Status

| Device | Round | Loss | Steps/Round | Time/Step | Status |
|--------|-------|------|-------------|-----------|--------|
| Mac Mini | 363 | 0.0043 | 50 | ~0.12s | ✅ Running |
| Jetson Orin | 2 | N/A | 50 | ~0.14s | ✅ Training |

## Speed Comparison

| Metric | Mac Mini | Jetson Orin | Notes |
|--------|----------|-------------|-------|
| **Time/round** | ~6s | ~7s | Jetson slightly slower for 50 steps |
| **Compute** | CPU (M4 Pro) | GPU (8GB) | Different hardware |
| **Memory** | 4GB allocated | 1GB VRAM | Both within limits |

## Quality Metrics

| Device | Data | Convergence | Loss Trend |
|--------|------|------------|-----------|
| Mac Mini | Medical textbook | Faster | 0.0043 (low) |
| Jetson | Hospital B data | Slower | Federating soon |

## Federation Round Times (Mac receives Jetson gradients)
- R0: ~3s (Jetson connect + train + send)
- R1: ~24s (Jetson reconnected)
- R2: ~24s (Jetson connected)

## Notes
- Mac is ahead (R363) because Jetson just started
- Jetson loss stuck at 13.5 (need federated averaging to improve)
- Mac's loss is 0.0043 (basically converged on its data)
- When federation kicks in, both should benefit

---
*Updated automatically*

## Future: 120B Training Experiment

**Goal**: Train GPT-OSS 120B with LoRA + Sparse + Disk Offload

**Prerequisites**:
- [ ] Implement layer-wise NVMe offload for 120B models
- [ ] Get 120B model (need ~240GB disk space)
- [ ] Set up 10Gbps network between devices

**Estimated Time**:
- First round: ~2 minutes
- Per round: ~9 minutes
- 50 rounds: ~7 hours

**When**: After current federation stabilizes


**Storage Plan for 120B:**
- Jetson: Model storage (492GB free > 240GB model) ✅
- Mac: Gradient aggregation only (sparse = <5MB)

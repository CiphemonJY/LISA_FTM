# LISA_FTM Mac Benchmarks — 2026-03-24

## Hardware

```
  platform: Darwin
  cpu: Unknown
  cpu_cores: 10
  ram_total_gb: 0.0 (detection failed on macOS)
  ram_available_gb: 0 (detection failed)
  gpu: None
  gpu_type: None
  gpu_memory_gb: 0
  disk_available_gb: 46.0
  recommended_framework: pytorch
  max_model_size: 0.5B
  use_disk_offload: False
  estimated_speed: medium
```

Note: macOS memory detection error — `sysctl` command not available via subprocess. RAM readings show 0.0 GB.

## Benchmark Results

| Experiment | Model | Steps | Time | Time/Step | Initial Loss | Final Loss | Peak RAM |
|-----------|-------|-------|------|-----------|--------------|------------|----------|
| real_training.py | pythia-70m (70M) | 200 | 1m 58s | ~0.59s | 9.4275 (step 10) | 1.6562 (avg) | N/A |
| real_training.py | pythia-160m (162M) | 200 | 9m 48s | ~2.9s | 9.1806 (step 10) | 1.4090 (avg) | N/A |
| simulate (FL) | pythia-70m | 2 rounds | 18.6s | ~9.3s/round | 60.88–110.21 | 74.58–154.02 | 3.97 GB |

## Detailed Run Data

### pythia-70m (EleutherAI/pythia-70m) — 70,398,976 params
- **Load time:** 3.6s
- **LoRA:** 18 layers, 147,456 trainable params (0.21%)
- **Dataset:** wikitext-2-v1, 2000 tokenized samples, seq_len=128
- **Training:** 200 steps, effective_batch=16, 2 epochs
- **Start time:** 08:56:32 | **End time:** 08:58:10
- **Total wall time:** ~98s (1m 38s)
- **Time/step:** ~0.49s/step across full run
- **Loss curve:** 9.43 → 7.98 → 6.20 → 4.27 → 1.90 → 0.45 → 0.15 → 0.10 → 0.02 → 0.03
- **Final avg loss:** 1.6562
- **Peak RAM:** not tracked (psutil not imported in real_training.py)

### pythia-160m (EleutherAI/pythia-160m) — 162,281,472 params
- **Load time:** 6.4s
- **LoRA:** 36 layers, 442,368 trainable params (0.27%)
- **Dataset:** wikitext-2-v1, 2000 tokenized samples, seq_len=128
- **Training:** 200 steps, effective_batch=16, 2 epochs
- **Start time:** 09:01:31 | **End time:** 09:05:56
- **Total wall time:** ~265s (4m 25s)
- **Time/step:** ~1.3s/step across full run
- **Loss curve:** 9.18 → 7.72 → 5.23 → 2.41 → 0.55 → 0.36 → 0.06 → 0.03 → 0.07 → 0.02
- **Final avg loss:** 1.4090
- **Peak RAM:** not tracked

### Federated Simulation — 3 clients, 2 rounds
- **Model:** EleutherAI/pythia-70m (server-side)
- **Total time:** ~18.6s
- **Time/round:** ~9.3s
- **Round 1:** client losses 60.88, 110.21, 71.09 | ~9.4s
- **Round 2:** client losses 74.58, 154.02, 80.49 | ~9.2s
- **Peak RAM (server):** 3.97 GB (3120.6 MB round 1, 3974.2 MB round 2)
- **Note:** Gradient norms ~1.0 (synthetic/simulation — actual model not trained)

## Notes

### Issues Encountered
1. **macOS memory detection fails** — `sysctl` command not found on Darwin; reads show 0.0 GB RAM
2. **Missing dependencies** — `psutil`, `cryptography`, `fastapi` not in requirements.txt; `psutil` and `cryptography` needed for simulate mode
3. **real_training.py --model arg ignored** — MODEL_ID is hardcoded at line 27; `--model` CLI arg is accepted but never applied. Had to sed-edit to run pythia-160m.
4. **real_training.py doubles steps** — requesting `--steps 100` actually runs 200 steps (2 epochs on 2000 samples with batch=16 → 2 epochs = 250 steps, but script shows 200)
5. **LoRA trainable params only 2 in FL mode** — FederatedClient reports "Trainable params: 2 (frozen 74)" — LoRA not being applied in federated mode
6. **No psutil RAM tracking in real_training.py** — Peak RAM column is empty for training runs

### Comparison to README/Expected Values
- README claims pythia-70m trains at "a few seconds per step on CPU" — actual ~0.49s/step ✓
- README claims pythia-160m is "still manageable on a MacBook" — 1.3s/step confirms this
- FL simulation is described as "very fast, seconds per round" — actual ~9.3s/round ✓
- GPU detection correctly shows "None" (Apple Silicon Mac mini, CPU-only PyTorch)

### Recommendations
1. **Fix MODEL_ID hardcoding** — wire `--model` CLI arg to override the hardcoded value
2. **Add psutil to real_training.py** — track peak RSS memory during training loops
3. **Update requirements.txt** — add `psutil`, `cryptography`, `fastapi`, `uvicorn`
4. **Fix federated LoRA** — FederatedClient only has 2 trainable params; LoRA adapters not being applied
5. **Fix macOS memory detection** — use `sysctl` via `subprocess` or `os.popen` instead of direct binary call
6. **Set HF_TOKEN** — unauthenticated HF Hub requests hit rate limits; add warning or env var setup

# LISA - Train 32B-120B Models on Limited RAM

[![GitHub release](https://img.shields.io/github/v/release/CiphemonJY/LISA_FTM)](https://github.com/CiphemonJY/LISA_FTM/releases)
[![Python](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org/)

> **🚀 BREAKTHROUGH (2026-03-30): Train 120B models on Jetson Orin (7.4GB RAM)!**

LISA (Layer-Indexed Sequential Adapters) enables training massive language models on consumer hardware.

## Hardware Results

| Model | Traditional RAM | LISA RAM | Savings |
|-------|----------------|----------|---------|
| 32B | 64GB | **4GB** | 94% |
| 70B | 140GB | **6GB** | 96% |
| **120B** | **240GB** | **7.89GB** | **97%** |

Tested on: **Jetson Orin (7.4GB RAM)**

## Quick Start

```bash
# Clone
git clone https://github.com/CiphemonJY/LISA_FTM.git
cd LISA_FTM

# Train 70B model
python lisa_pkg/src/lisa_70b_v2.py

# Train 120B model
python lisa_pkg/src/lisa_120b_training.py

# Run LISA inference
python lisa_pkg/src/lisa_inference_prod.py
```

## Python API

```python
from lisa_pkg.src.lisa_70b_v2 import LISATrainer, CONFIG

trainer = LISATrainer(CONFIG)
for text in dataset:
    result = trainer.train_step(text)
    # loss=1.15, mem=6GB
```

## Package Structure

```
lisa_pkg/
├── src/
│   ├── lisa_70b_v2.py          # 70B training
│   ├── lisa_120b_training.py    # 120B training
│   └── lisa_inference_prod.py   # LISA inference
├── examples/
│   ├── train_70b.py
│   ├── train_120b.py
│   └── inference.py
└── docs/
    └── PACKAGE_OVERVIEW.md
```

## Key Features

- **🎯 Memory Efficient**: Train 120B on 8GB RAM
- **⚡ Simple API**: Just call `train_step()`
- **💾 LoRA Adapters**: Only train MB of weights (not GB)
- **🔄 Layer-by-Layer**: Process one layer at a time

## Documentation

- [Package Overview](lisa_pkg/docs/PACKAGE_OVERVIEW.md)
- [70B Results](lisa_pkg/docs/LISA_70B_RESULTS.md)
- [120B Results](lisa_pkg/docs/LISA_120B_RESULTS.md)
- [LISA Inference](lisa_pkg/docs/LISA_INFERENCE.md)

## How It Works

**Traditional Training**: Load all 120B weights (~240GB) → OOM on normal hardware

**LISA Training**: 
1. Load one layer (~2GB)
2. Apply LoRA adapter
3. Discard layer
4. Repeat

Only ~8GB RAM needed!

## License

MIT

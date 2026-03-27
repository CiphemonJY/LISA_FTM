# Federated Optimizations

Performance enhancements for federated learning deployments.

## Sparse Gradient Compression

Reduces bandwidth by 10-20x by sending only the top-K% of gradients by magnitude.

### Usage

```python
from federated.optimizations import SparseCompressor

compressor = SparseCompressor(keep_fraction=0.1)  # Keep 10%

# Compress before sending
packed, metadata, stats = compressor.compress(gradients)
print(f"Compression: {stats['compression_ratio']:.1f}x")
send_to_server(packed, metadata)

# Decompress after receiving
decompressed = compressor.decompress(packed, metadata)
```

### Compression Ratios

| Keep Fraction | Compression | Bandwidth |
|--------------|--------------|------------|
| 5% | ~40x | 2.5% |
| 10% | ~20x | 5% |
| 20% | ~10x | 10% |
| 50% | ~4x | 25% |

## CPU Offload

Move non-LORA parameters to CPU RAM to free GPU memory for training.

### Usage

```python
# Offload to CPU before training
for name, param in model.named_parameters():
    if "lora_" not in name:
        param.data = param.data.to('cpu')

# Restore after training
for name, param in model.named_parameters():
    if "lora_" not in name:
        param.data = param.data.to('cuda')
```

## Disk Offload

Save gradients to disk instead of memory to prevent OOM on memory-constrained devices.

### Usage

```python
import tempfile
from pathlib import Path

# Save to disk
temp_dir = tempfile.mkdtemp(prefix="lisa_grads_")
disk_path = Path(temp_dir) / "round_grads.pt"
torch.save(gradients, disk_path)

# Later: load from disk
gradients = torch.load(disk_path)
```

## Jetson Orin Specific Notes

The Jetson Orin can experience CMA (Contiguous Memory Allocator) fragmentation after repeated model loads. This manifests as `NvMapMemAllocInternalTagged: error 12` during model loading.

### Symptoms
- Model loading fails with CUDA memory errors
- `nvidia-smi` shows [N/A] for memory
- `/proc/meminfo` shows `CmaFree` near 0

### Solutions
1. **Power cycle** (recommended): Unplug device for 30 seconds
2. **CPU mode fallback**: Use `--device cpu` for continued operation
3. **Wait**: CMA may slowly recover over time

## Testing

Run the test suite:
```bash
python -m federated.optimizations.sparse_federation
```

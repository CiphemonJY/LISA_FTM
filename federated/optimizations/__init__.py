"""
Federated Optimizations Package
==============================
Performance enhancements for federated learning.

Includes:
- Sparse gradient compression (10-20x bandwidth reduction)
- CPU offload for memory-constrained devices
- Disk offload for large gradient storage
"""
from .sparse_federation import SparseCompressor

__all__ = ['SparseCompressor']

"""Compatibility loader for chunked N8 cache directories.

This helper is intentionally small; the canonical training script should use
train_dit_v3.py directly once its load_cache implementation is updated.
"""

from .train_dit_v3 import load_cache, next_cached_batch

__all__ = ["load_cache", "next_cached_batch"]

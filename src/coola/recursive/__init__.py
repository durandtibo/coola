"""Recursive data transformation using DFS pattern.

This design is inspired by the DFS array iterator pattern and provides:
1. Memory-efficient generator-based traversal
2. Clean separation between transformation logic and type dispatch
3. No state object threading through recursion
4. Easy extensibility via registry pattern
"""

from __future__ import annotations

__all__ = [
    "TransformerRegistry",
    "get_default_registry",
    "recursive_apply",
    "register_transformers",
]

from batcharray.recursive2.interface import (
    get_default_registry,
    recursive_apply,
    register_transformers,
)
from batcharray.recursive2.registry import TransformerRegistry

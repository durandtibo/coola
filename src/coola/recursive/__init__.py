"""Recursive data transformation using DFS pattern.

This design is inspired by the DFS array iterator pattern and provides:
1. Memory-efficient generator-based traversal
2. Clean separation between transformation logic and type dispatch
3. No state object threading through recursion
4. Easy extensibility via registry pattern
"""

from __future__ import annotations

__all__ = [
    "BaseTransformer",
    "ConditionalTransformer",
    "DefaultTransformer",
    "MappingTransformer",
    "SequenceTransformer",
    "SetTransformer",
    "TransformerRegistry",
    "get_default_registry",
    "recursive_apply",
    "register_transformers",
]

from coola.recursive.base import BaseTransformer
from coola.recursive.conditional import ConditionalTransformer
from coola.recursive.default import DefaultTransformer
from coola.recursive.interface import (
    get_default_registry,
    recursive_apply,
    register_transformers,
)
from coola.recursive.mapping import MappingTransformer
from coola.recursive.registry import TransformerRegistry
from coola.recursive.sequence import SequenceTransformer
from coola.recursive.set import SetTransformer

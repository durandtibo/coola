r"""Depth-first traversal utilities for nested data structures."""

from __future__ import annotations

__all__ = [
    "BaseIterator",
    "DefaultIterator",
    "IterableIterator",
    "IteratorRegistry",
    "MappingIterator",
    "dfs_iterate",
    "get_default_registry",
    "register_iterators",
]

from coola.iterator.dfs.base import BaseIterator
from coola.iterator.dfs.default import DefaultIterator
from coola.iterator.dfs.interface import (
    dfs_iterate,
    get_default_registry,
    register_iterators,
)
from coola.iterator.dfs.iterable import IterableIterator
from coola.iterator.dfs.mapping import MappingIterator
from coola.iterator.dfs.registry import IteratorRegistry

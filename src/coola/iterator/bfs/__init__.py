r"""Contain code to iterate over nested data with a Breath-First Search
(BFS) strategy."""

from __future__ import annotations

__all__ = [
    "BaseChildFinder",
    "ChildFinderRegistry",
    "DefaultChildFinder",
    "IterableChildFinder",
    "MappingChildFinder",
    "bfs_iterate",
    "get_default_registry",
    "register_child_finders",
]

from coola.iterator.bfs.base import BaseChildFinder
from coola.iterator.bfs.default import DefaultChildFinder
from coola.iterator.bfs.interface import (
    bfs_iterate,
    get_default_registry,
    register_child_finders,
)
from coola.iterator.bfs.iterable import IterableChildFinder
from coola.iterator.bfs.mapping import MappingChildFinder
from coola.iterator.bfs.registry import ChildFinderRegistry

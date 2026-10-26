r"""Utilities to traverse and filter nested data structures.

``dfs_iterate`` and ``bfs_iterate`` are lazy generators that yield the leaves of a nested
structure, once per occurrence and never a container, in depth-first (pre-order) or breadth-first
(level order) order. The child finder of an object is chosen from its type along the Method
Resolution Order, and one child finder serves both orders. Keys are never yielded, the input is
never modified, iterators and generators are leaves (never consumed), and a container that
contains itself raises a ``ValueError``. Both orders yield the same leaves; only the order
differs, and neither is limited by the depth of the data.
"""

from __future__ import annotations

__all__ = [
    "BaseChildFinder",
    "ChildFinder",
    "ChildFinderRegistry",
    "DataclassChildFinder",
    "DefaultChildFinder",
    "IterableChildFinder",
    "MappingChildFinder",
    "bfs_iterate",
    "dfs_iterate",
    "filter_by_type",
    "get_default_child_finder_registry",
    "get_default_registry",
    "register_child_finders",
]

from coola.iterator.base import BaseChildFinder, ChildFinder
from coola.iterator.dataclass import DataclassChildFinder
from coola.iterator.default import DefaultChildFinder
from coola.iterator.filtering import filter_by_type
from coola.iterator.interface import (
    bfs_iterate,
    dfs_iterate,
    get_default_child_finder_registry,
    get_default_registry,
    register_child_finders,
)
from coola.iterator.iterable import IterableChildFinder
from coola.iterator.mapping import MappingChildFinder
from coola.iterator.registry import ChildFinderRegistry

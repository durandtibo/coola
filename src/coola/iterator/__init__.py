r"""Utilities to traverse and filter nested data structures.

``dfs_iterate`` and ``bfs_iterate`` are lazy generators that yield the leaves of a nested
structure, once per occurrence and never a container, in depth-first (pre-order) or breadth-first
(level order) order. The iterator is chosen from the type of the data along the Method Resolution
Order. Keys are never yielded, the input is never modified, iterators and generators are leaves
(never consumed), and a container that contains itself raises a ``ValueError``. Both orders
yield the same leaves; only the order differs.
"""

from __future__ import annotations

__all__ = ["bfs_iterate", "dfs_iterate", "filter_by_type"]

from coola.iterator.bfs import bfs_iterate
from coola.iterator.dfs import dfs_iterate
from coola.iterator.filtering import filter_by_type

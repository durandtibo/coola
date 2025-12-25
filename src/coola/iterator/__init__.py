r"""Contain code to iterate over nested data."""

from __future__ import annotations

__all__ = ["bfs_iterate", "dfs_iterate", "filter_by_type"]

from coola.iterator.bfs import bfs_iterate
from coola.iterator.dfs import dfs_iterate
from coola.iterator.filtering import filter_by_type

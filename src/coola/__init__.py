r"""Contain the main features of the ``coola`` package."""

from __future__ import annotations

__all__ = [
    "__version__",
    "bfs_iterate",
    # Nested data utilities
    "convert_to_dict_of_lists",
    "convert_to_list_of_dicts",
    # Iteration
    "dfs_iterate",
    "filter_by_type",
    "objects_are_allclose",
    # Equality checking functions
    "objects_are_equal",
    # Recursive transformation
    "recursive_apply",
    # Summarization
    "summarize",
    "to_flat_dict",
]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed, fallback if needed
    __version__ = "0.0.0"

# Import main user-facing functions for convenience
# These are the most commonly used functions in the library
from coola.equality import objects_are_allclose, objects_are_equal
from coola.iterator import bfs_iterate, dfs_iterate, filter_by_type
from coola.nested import (
    convert_to_dict_of_lists,
    convert_to_list_of_dicts,
    to_flat_dict,
)
from coola.recursive import recursive_apply
from coola.summary import summarize

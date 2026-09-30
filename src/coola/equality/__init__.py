r"""Utilities to compare nested objects for exact or tolerant
equality."""

from __future__ import annotations

__all__ = [
    "ComparisonResult",
    "assert_objects_equal",
    "compare",
    "objects_are_allclose",
    "objects_are_equal",
]

from coola.equality.interface import objects_are_allclose, objects_are_equal
from coola.equality.result import ComparisonResult, assert_objects_equal, compare

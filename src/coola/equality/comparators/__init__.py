r"""Contain the comparators to check if two objects are equal or not."""

from __future__ import annotations

__all__ = [
    "ArrayEqualityComparator",
    "BaseEqualityComparator",
    "DefaultEqualityComparator",
]

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.comparators.default import DefaultEqualityComparator
from coola.equality.comparators.numpy_ import ArrayEqualityComparator

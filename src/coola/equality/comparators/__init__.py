r"""Contain the comparators to check if two objects are equal or not."""

from __future__ import annotations

__all__ = [
    "ArrayEqualityComparator",
    "BaseEqualityComparator",
    "DefaultEqualityComparator",
    "TensorEqualityComparator",
    "get_type_comparator_mapping",
]

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.comparators.default import DefaultEqualityComparator
from coola.equality.comparators.numpy_ import ArrayEqualityComparator
from coola.equality.comparators.torch_ import TensorEqualityComparator
from coola.equality.comparators.utils import get_type_comparator_mapping
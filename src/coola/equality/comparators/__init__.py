r"""Contain the comparators to check if two objects are equal or not."""

from __future__ import annotations

__all__ = [
    "BaseEqualityComparator",
    "DefaultEqualityComparator",
    "JaxArrayEqualityComparator",
    "MappingEqualityComparator",
    "NumpyArrayEqualityComparator",
    "SequenceEqualityComparator",
    "TorchPackedSequenceEqualityComparator",
    "TorchTensorEqualityComparator",
    "XarrayDatasetEqualityComparator",
    "XarrayVariableEqualityComparator",
    "get_type_comparator_mapping",
]

from coola.equality.comparators.base import BaseEqualityComparator
from coola.equality.comparators.collection import (
    MappingEqualityComparator,
    SequenceEqualityComparator,
)
from coola.equality.comparators.default import DefaultEqualityComparator
from coola.equality.comparators.jax_ import JaxArrayEqualityComparator
from coola.equality.comparators.numpy_ import NumpyArrayEqualityComparator
from coola.equality.comparators.torch_ import (
    TorchPackedSequenceEqualityComparator,
    TorchTensorEqualityComparator,
)
from coola.equality.comparators.utils import get_type_comparator_mapping
from coola.equality.comparators.xarray_ import (
    XarrayDatasetEqualityComparator,
    XarrayVariableEqualityComparator,
)

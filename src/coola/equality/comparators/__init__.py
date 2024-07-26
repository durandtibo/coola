r"""Contain the comparators to check if two objects are equal or not."""

from __future__ import annotations

__all__ = [
    "BaseEqualityComparator",
    "DefaultEqualityComparator",
    "JaxArrayEqualityComparator",
    "MappingEqualityComparator",
    "NumpyArrayEqualityComparator",
    "NumpyMaskedArrayEqualityComparator",
    "PandasDataFrameEqualityComparator",
    "PandasSeriesEqualityComparator",
    "PolarsDataFrameEqualityComparator",
    "PolarsSeriesEqualityComparator",
    "PyarrowEqualityComparator",
    "ScalarEqualityComparator",
    "SequenceEqualityComparator",
    "TorchPackedSequenceEqualityComparator",
    "TorchTensorEqualityComparator",
    "XarrayDataArrayEqualityComparator",
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
from coola.equality.comparators.numpy_ import (
    NumpyArrayEqualityComparator,
    NumpyMaskedArrayEqualityComparator,
)
from coola.equality.comparators.pandas_ import (
    PandasDataFrameEqualityComparator,
    PandasSeriesEqualityComparator,
)
from coola.equality.comparators.polars_ import (
    PolarsDataFrameEqualityComparator,
    PolarsSeriesEqualityComparator,
)
from coola.equality.comparators.pyarrow_ import PyarrowEqualityComparator
from coola.equality.comparators.scalar import ScalarEqualityComparator
from coola.equality.comparators.torch_ import (
    TorchPackedSequenceEqualityComparator,
    TorchTensorEqualityComparator,
)
from coola.equality.comparators.utils import get_type_comparator_mapping
from coola.equality.comparators.xarray_ import (
    XarrayDataArrayEqualityComparator,
    XarrayDatasetEqualityComparator,
    XarrayVariableEqualityComparator,
)

r"""Contain the equality testers to check if two objects are equal or
not."""

from __future__ import annotations

__all__ = [
    "BaseEqualityTester",
    "DefaultEqualityTester",
    "EqualEqualityTester",
    "EqualNanEqualityTester",
    "EqualityTesterRegistry",
    "JaxArrayEqualityTester",
    "MappingEqualityTester",
    "NumpyArrayEqualityTester",
    "NumpyMaskedArrayEqualityTester",
    "PandasDataFrameEqualityTester",
    "PandasSeriesEqualityTester",
    "PolarsDataFrameEqualityTester",
    "PolarsLazyFrameEqualityTester",
    "PolarsSeriesEqualityTester",
    "PyarrowEqualityTester",
    "ScalarEqualityTester",
    "SequenceEqualityTester",
    "TorchPackedSequenceEqualityTester",
    "TorchTensorEqualityTester",
    "XarrayDataArrayEqualityTester",
    "XarrayDatasetEqualityTester",
    "XarrayVariableEqualityTester",
    "get_default_registry",
    "register_equality_testers",
]

from coola.equality.tester.base import BaseEqualityTester
from coola.equality.tester.collection import (
    MappingEqualityTester,
    SequenceEqualityTester,
)
from coola.equality.tester.default import DefaultEqualityTester
from coola.equality.tester.equal import EqualEqualityTester, EqualNanEqualityTester
from coola.equality.tester.interface import (
    get_default_registry,
    register_equality_testers,
)
from coola.equality.tester.jax import JaxArrayEqualityTester
from coola.equality.tester.numpy import (
    NumpyArrayEqualityTester,
    NumpyMaskedArrayEqualityTester,
)
from coola.equality.tester.pandas import (
    PandasDataFrameEqualityTester,
    PandasSeriesEqualityTester,
)
from coola.equality.tester.polars import (
    PolarsDataFrameEqualityTester,
    PolarsLazyFrameEqualityTester,
    PolarsSeriesEqualityTester,
)
from coola.equality.tester.pyarrow import PyarrowEqualityTester
from coola.equality.tester.registry import EqualityTesterRegistry
from coola.equality.tester.scalar import ScalarEqualityTester
from coola.equality.tester.torch import (
    TorchPackedSequenceEqualityTester,
    TorchTensorEqualityTester,
)
from coola.equality.tester.xarray import (
    XarrayDataArrayEqualityTester,
    XarrayDatasetEqualityTester,
    XarrayVariableEqualityTester,
)

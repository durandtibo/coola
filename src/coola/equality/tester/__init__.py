r"""Type-aware equality testers used by ``coola.equality``.

This package exposes a registry-based dispatch system with built-in testers
for common Python containers and optional scientific types. Extend the
default behavior by registering custom testers in the default registry.

Example:
    ```pycon
    >>> from coola.equality.tester import get_default_registry
    >>> from coola.equality.config import EqualityConfig
    >>> registry = get_default_registry()
    >>> config = EqualityConfig()
    >>> # Compare lists with nested structures
    >>> registry.objects_are_equal([1, {"a": 2}], [1, {"a": 2}], config)
    True

    ```
"""

from __future__ import annotations

__all__ = [
    "BaseEqualityTester",
    "DefaultEqualityTester",
    "EqualEqualityTester",
    "EqualNanEqualityTester",
    "EqualityTesterRegistry",
    "HandlerEqualityTester",
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
    "TolerantEqualEqualityTester",
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
from coola.equality.tester.handler import HandlerEqualityTester
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
from coola.equality.tester.tolerant import TolerantEqualEqualityTester
from coola.equality.tester.torch import (
    TorchPackedSequenceEqualityTester,
    TorchTensorEqualityTester,
)
from coola.equality.tester.xarray import (
    XarrayDataArrayEqualityTester,
    XarrayDatasetEqualityTester,
    XarrayVariableEqualityTester,
)

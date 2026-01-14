r"""Contain the equality testers to check if two objects are equal or
not.

This module provides a comprehensive system for comparing objects of different types
using a registry-based approach with specialized equality testers. The testers use
the chain of responsibility pattern with handlers to perform various equality checks.

Key components:
    - BaseEqualityTester: Abstract base class for all equality testers
    - EqualityTesterRegistry: Registry that dispatches to appropriate testers by type
    - Specialized testers for Python built-ins (list, dict, int, float, etc.)
    - Testers for third-party libraries (NumPy, PyTorch, Pandas, Polars, JAX, xarray, PyArrow)

The default registry is pre-configured with testers for common types and can be
extended with custom testers using register_equality_testers().

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

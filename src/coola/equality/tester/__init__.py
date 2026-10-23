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
    "BaseChainEqualityTester",
    "BaseEqualityTester",
    "DataclassEqualityTester",
    "DefaultEqualityTester",
    "EqualEqualityTester",
    "EqualNanEqualityTester",
    "EqualityTesterRegistry",
    "HandlerEqualityTester",
    "JaxArrayEqualityTester",
    "MappingEqualityTester",
    "NumpyArrayEqualityTester",
    "NumpyMaskedArrayEqualityTester",
    "OrderedMappingEqualityTester",
    "PandasDataFrameEqualityTester",
    "PandasExtensionArrayEqualityTester",
    "PandasIndexEqualityTester",
    "PandasSeriesEqualityTester",
    "PolarsDataFrameEqualityTester",
    "PolarsLazyFrameEqualityTester",
    "PolarsSeriesEqualityTester",
    "PyarrowEqualityTester",
    "PydanticModelEqualityTester",
    "ScalarEqualityTester",
    "SequenceEqualityTester",
    "TolerantEqualEqualityTester",
    "TorchPackedSequenceEqualityTester",
    "TorchTensorEqualityTester",
    "XarrayDataArrayEqualityTester",
    "XarrayDatasetEqualityTester",
    "XarrayVariableEqualityTester",
    "get_default_equality_tester_registry",
    "get_default_registry",
    "register_equality_testers",
]

from typing import TYPE_CHECKING

from coola.equality.tester.base import BaseEqualityTester
from coola.equality.tester.chain import BaseChainEqualityTester
from coola.equality.tester.collection import (
    MappingEqualityTester,
    OrderedMappingEqualityTester,
    SequenceEqualityTester,
)
from coola.equality.tester.dataclass import DataclassEqualityTester
from coola.equality.tester.default import DefaultEqualityTester
from coola.equality.tester.equal import EqualEqualityTester, EqualNanEqualityTester
from coola.equality.tester.handler import HandlerEqualityTester
from coola.equality.tester.interface import (
    get_default_equality_tester_registry,
    get_default_registry,
    register_equality_testers,
)
from coola.equality.tester.registry import EqualityTesterRegistry
from coola.equality.tester.scalar import ScalarEqualityTester
from coola.equality.tester.tolerant import TolerantEqualEqualityTester
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.equality.tester.jax import JaxArrayEqualityTester
    from coola.equality.tester.numpy import (
        NumpyArrayEqualityTester,
        NumpyMaskedArrayEqualityTester,
    )
    from coola.equality.tester.pandas import (
        PandasDataFrameEqualityTester,
        PandasExtensionArrayEqualityTester,
        PandasIndexEqualityTester,
        PandasSeriesEqualityTester,
    )
    from coola.equality.tester.polars import (
        PolarsDataFrameEqualityTester,
        PolarsLazyFrameEqualityTester,
        PolarsSeriesEqualityTester,
    )
    from coola.equality.tester.pyarrow import PyarrowEqualityTester
    from coola.equality.tester.pydantic import PydanticModelEqualityTester
    from coola.equality.tester.torch import (
        TorchPackedSequenceEqualityTester,
        TorchTensorEqualityTester,
    )
    from coola.equality.tester.xarray import (
        XarrayDataArrayEqualityTester,
        XarrayDatasetEqualityTester,
        XarrayVariableEqualityTester,
    )


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "JaxArrayEqualityTester": "coola.equality.tester.jax",
    "NumpyArrayEqualityTester": "coola.equality.tester.numpy",
    "NumpyMaskedArrayEqualityTester": "coola.equality.tester.numpy",
    "PandasDataFrameEqualityTester": "coola.equality.tester.pandas",
    "PandasExtensionArrayEqualityTester": "coola.equality.tester.pandas",
    "PandasIndexEqualityTester": "coola.equality.tester.pandas",
    "PandasSeriesEqualityTester": "coola.equality.tester.pandas",
    "PolarsDataFrameEqualityTester": "coola.equality.tester.polars",
    "PolarsLazyFrameEqualityTester": "coola.equality.tester.polars",
    "PolarsSeriesEqualityTester": "coola.equality.tester.polars",
    "PydanticModelEqualityTester": "coola.equality.tester.pydantic",
    "PyarrowEqualityTester": "coola.equality.tester.pyarrow",
    "TorchPackedSequenceEqualityTester": "coola.equality.tester.torch",
    "TorchTensorEqualityTester": "coola.equality.tester.torch",
    "XarrayDataArrayEqualityTester": "coola.equality.tester.xarray",
    "XarrayDatasetEqualityTester": "coola.equality.tester.xarray",
    "XarrayVariableEqualityTester": "coola.equality.tester.xarray",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())

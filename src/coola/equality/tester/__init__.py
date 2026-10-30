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
    "DataclassEqualityTester",
    "DefaultEqualityTester",
    "DifferenceCloseEqualityTester",
    "EqualEqualityTester",
    "EqualNanEqualityTester",
    "EqualityTesterRegistry",
    "JaxArrayEqualityTester",
    "MappingEqualityTester",
    "NumpyArrayEqualityTester",
    "NumpyMaskedArrayEqualityTester",
    "NumpyTimeScalarEqualityTester",
    "ObjectAttributesEqualityTester",
    "OrderedMappingEqualityTester",
    "PydanticModelEqualityTester",
    "ScalarEqualityTester",
    "ScipySparseEqualityTester",
    "SequenceEqualityTester",
    "SetEqualityTester",
    "TableEqualityTester",
    "TolerantEqualEqualityTester",
    "TorchPackedSequenceEqualityTester",
    "TorchTensorEqualityTester",
    "XarrayDataArrayEqualityTester",
    "XarrayDatasetEqualityTester",
    "XarrayVariableEqualityTester",
    "get_default_registry",
    "register_equality_testers",
]

from typing import TYPE_CHECKING

from coola.equality.tester.base import BaseEqualityTester
from coola.equality.tester.collection import (
    MappingEqualityTester,
    OrderedMappingEqualityTester,
    SequenceEqualityTester,
    SetEqualityTester,
)
from coola.equality.tester.dataclass import DataclassEqualityTester
from coola.equality.tester.default import DefaultEqualityTester
from coola.equality.tester.difference import DifferenceCloseEqualityTester
from coola.equality.tester.equal import EqualEqualityTester, EqualNanEqualityTester
from coola.equality.tester.interface import (
    get_default_registry,
    register_equality_testers,
)
from coola.equality.tester.object import ObjectAttributesEqualityTester
from coola.equality.tester.registry import EqualityTesterRegistry
from coola.equality.tester.scalar import ScalarEqualityTester
from coola.equality.tester.table import TableEqualityTester
from coola.equality.tester.tolerant import TolerantEqualEqualityTester
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.equality.tester.jax import JaxArrayEqualityTester
    from coola.equality.tester.numpy import (
        NumpyArrayEqualityTester,
        NumpyMaskedArrayEqualityTester,
        NumpyTimeScalarEqualityTester,
    )
    from coola.equality.tester.pydantic import PydanticModelEqualityTester
    from coola.equality.tester.scipy import ScipySparseEqualityTester
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
    "NumpyTimeScalarEqualityTester": "coola.equality.tester.numpy",
    "PydanticModelEqualityTester": "coola.equality.tester.pydantic",
    "ScipySparseEqualityTester": "coola.equality.tester.scipy",
    "TorchPackedSequenceEqualityTester": "coola.equality.tester.torch",
    "TorchTensorEqualityTester": "coola.equality.tester.torch",
    "XarrayDataArrayEqualityTester": "coola.equality.tester.xarray",
    "XarrayDatasetEqualityTester": "coola.equality.tester.xarray",
    "XarrayVariableEqualityTester": "coola.equality.tester.xarray",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())

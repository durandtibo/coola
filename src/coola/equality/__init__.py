r"""Utilities to compare nested objects for exact or tolerant equality.

The package exports the comparison functions, the result, the
configuration, the engine, the testers and the registry. The testers of
the optional libraries (NumPy, PyTorch, JAX, xarray, pydantic) are
imported the first time they are used.
"""

from __future__ import annotations

__all__ = [
    "AttributeName",
    "BaseEqualityTester",
    "ComparisonResult",
    "DataclassEqualityTester",
    "DefaultEqualityTester",
    "DifferenceCloseEqualityTester",
    "Element",
    "Engine",
    "EqualEqualityTester",
    "EqualNanEqualityTester",
    "EqualityConfig",
    "EqualityTesterRegistry",
    "JaxArrayEqualityTester",
    "MappingEqualityTester",
    "MaxDepthExceededError",
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
    "Tolerance",
    "TolerantEqualEqualityTester",
    "TorchPackedSequenceEqualityTester",
    "TorchTensorEqualityTester",
    "XarrayDataArrayEqualityTester",
    "XarrayDatasetEqualityTester",
    "XarrayVariableEqualityTester",
    "assert_objects_allclose",
    "assert_objects_equal",
    "compare",
    "get_default_registry",
    "objects_are_allclose",
    "objects_are_equal",
    "register_equality_testers",
]

from typing import TYPE_CHECKING

from coola.equality._helpers.utils import AttributeName, MaxDepthExceededError
from coola.equality.config import EqualityConfig
from coola.equality.engine import Element, Engine
from coola.equality.interface import objects_are_allclose, objects_are_equal
from coola.equality.result import (
    ComparisonResult,
    assert_objects_allclose,
    assert_objects_equal,
    compare,
)
from coola.equality.tester import (
    BaseEqualityTester,
    DataclassEqualityTester,
    DefaultEqualityTester,
    DifferenceCloseEqualityTester,
    EqualEqualityTester,
    EqualityTesterRegistry,
    EqualNanEqualityTester,
    MappingEqualityTester,
    ObjectAttributesEqualityTester,
    OrderedMappingEqualityTester,
    ScalarEqualityTester,
    SequenceEqualityTester,
    SetEqualityTester,
    TableEqualityTester,
    TolerantEqualEqualityTester,
    get_default_registry,
    register_equality_testers,
)
from coola.equality.tolerance import Tolerance
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.equality.tester import (
        JaxArrayEqualityTester,
        NumpyArrayEqualityTester,
        NumpyMaskedArrayEqualityTester,
        NumpyTimeScalarEqualityTester,
        PydanticModelEqualityTester,
        ScipySparseEqualityTester,
        TorchPackedSequenceEqualityTester,
        TorchTensorEqualityTester,
        XarrayDataArrayEqualityTester,
        XarrayDatasetEqualityTester,
        XarrayVariableEqualityTester,
    )

# The testers of the optional libraries are imported the first time they are used, so the
# libraries are not imported with the package.
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

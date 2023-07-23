__all__ = [
    "BaseAllCloseOperator",
    "BaseEqualityOperator",
    "DataArrayAllCloseOperator",
    "DataArrayEqualityOperator",
    "DatasetAllCloseOperator",
    "DatasetEqualityOperator",
    "DefaultAllCloseOperator",
    "DefaultEqualityOperator",
    "MappingAllCloseOperator",
    "MappingEqualityOperator",
    "NDArrayAllCloseOperator",
    "NDArrayEqualityOperator",
    "PackedSequenceAllCloseOperator",
    "PackedSequenceEqualityOperator",
    "ScalarAllCloseOperator",
    "SequenceAllCloseOperator",
    "SequenceEqualityOperator",
    "TensorAllCloseOperator",
    "TensorEqualityOperator",
    "VariableAllCloseOperator",
    "VariableEqualityOperator",
    "get_mapping_allclose",
    "get_mapping_equality",
]

from coola.comparators import pandas_, polars_  # noqa: F401
from coola.comparators.allclose import (
    DefaultAllCloseOperator,
    MappingAllCloseOperator,
    ScalarAllCloseOperator,
    SequenceAllCloseOperator,
)
from coola.comparators.base import BaseAllCloseOperator, BaseEqualityOperator
from coola.comparators.equality import (
    DefaultEqualityOperator,
    MappingEqualityOperator,
    SequenceEqualityOperator,
)
from coola.comparators.numpy_ import NDArrayAllCloseOperator, NDArrayEqualityOperator
from coola.comparators.torch_ import (
    PackedSequenceAllCloseOperator,
    PackedSequenceEqualityOperator,
    TensorAllCloseOperator,
    TensorEqualityOperator,
)
from coola.comparators.utils import get_mapping_allclose, get_mapping_equality
from coola.comparators.xarray_ import (
    DataArrayAllCloseOperator,
    DataArrayEqualityOperator,
    DatasetAllCloseOperator,
    DatasetEqualityOperator,
    VariableAllCloseOperator,
    VariableEqualityOperator,
)


def register_allclose() -> None:
    r"""Register allclose operators to ``AllCloseTester``."""
    from coola.testers.allclose import AllCloseTester

    for typ, op in get_mapping_allclose().items():
        if not AllCloseTester.has_operator(typ):  # pragma: no cover
            AllCloseTester.add_operator(typ, op)


def register_equality() -> None:
    r"""Register equality operators to ``EqualityTester``."""
    from coola.testers.equality import EqualityTester

    for typ, op in get_mapping_equality().items():
        if not EqualityTester.has_operator(typ):  # pragma: no cover
            EqualityTester.add_operator(typ, op)


register_allclose()
register_equality()

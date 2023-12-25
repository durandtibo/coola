r"""This package contains the comparator implementations."""

__all__ = [
    "ArrayAllCloseOperator",
    "ArrayEqualityOperator",
    "BaseAllCloseOperator",
    "BaseEqualityOperator",
    "DataArrayAllCloseOperator",
    "DataArrayEqualityOperator",
    "DatasetAllCloseOperator",
    "DatasetEqualityOperator",
    "DefaultAllCloseOperator",
    "DefaultEqualityOperator",
    "JaxArrayAllCloseOperator",
    "JaxArrayEqualityOperator",
    "MappingAllCloseOperator",
    "MappingEqualityOperator",
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
from coola.comparators.jax_ import JaxArrayAllCloseOperator, JaxArrayEqualityOperator
from coola.comparators.numpy_ import ArrayAllCloseOperator, ArrayEqualityOperator
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
    r"""Register allclose operators to ``AllCloseTester``.

    ```pycon
    >>> from coola.comparators import register_allclose
    >>> from coola.testers import AllCloseTester
    >>> register_allclose()
    >>> tester = AllCloseTester()
    >>> tester
    AllCloseTester(
      (<class 'collections.abc.Mapping'>): MappingAllCloseOperator()
      (<class 'collections.abc.Sequence'>): SequenceAllCloseOperator()
      (<class 'bool'>): ScalarAllCloseOperator()
      (<class 'dict'>): MappingAllCloseOperator()
      (<class 'float'>): ScalarAllCloseOperator()
      (<class 'int'>): ScalarAllCloseOperator()
      (<class 'list'>): SequenceAllCloseOperator()
      (<class 'object'>): DefaultAllCloseOperator()
      (<class 'tuple'>): SequenceAllCloseOperator()
      (<class 'jax.Array'>): JaxArrayAllCloseOperator(check_dtype=True)
      (<class 'jaxlib.xla_extension.ArrayImpl'>): JaxArrayAllCloseOperator(check_dtype=True)
      (<class 'numpy.ndarray'>): ArrayAllCloseOperator(check_dtype=True)
      (<class 'pandas.core.frame.DataFrame'>): DataFrameAllCloseOperator()
      (<class 'pandas.core.series.Series'>): SeriesAllCloseOperator()
      (<class 'polars.dataframe.frame.DataFrame'>): DataFrameAllCloseOperator()
      (<class 'polars.series.series.Series'>): SeriesAllCloseOperator()
      (<class 'torch.Tensor'>): TensorAllCloseOperator()
      (<class 'torch.nn.utils.rnn.PackedSequence'>): PackedSequenceAllCloseOperator()
      (<class 'xarray.core.dataset.Dataset'>): DatasetAllCloseOperator()
      (<class 'xarray.core.dataarray.DataArray'>): DataArrayAllCloseOperator()
      (<class 'xarray.core.variable.Variable'>): VariableAllCloseOperator()
    )

    ```
    """
    from coola.testers.allclose import AllCloseTester

    for typ, op in get_mapping_allclose().items():
        if not AllCloseTester.has_operator(typ):  # pragma: no cover
            AllCloseTester.add_operator(typ, op)


def register_equality() -> None:
    r"""Register equality operators to ``EqualityTester``.

    >>> from coola.comparators import register_equality
    >>> from coola.testers import EqualityTester
    >>> register_equality()
    >>> tester = EqualityTester()
    >>> tester
    EqualityTester(
      (<class 'collections.abc.Mapping'>): MappingEqualityOperator()
      (<class 'collections.abc.Sequence'>): SequenceEqualityOperator()
      (<class 'dict'>): MappingEqualityOperator()
      (<class 'list'>): SequenceEqualityOperator()
      (<class 'object'>): DefaultEqualityOperator()
      (<class 'tuple'>): SequenceEqualityOperator()
      (<class 'jax.Array'>): JaxArrayEqualityOperator(check_dtype=True)
      (<class 'jaxlib.xla_extension.ArrayImpl'>): JaxArrayEqualityOperator(check_dtype=True)
      (<class 'numpy.ndarray'>): ArrayEqualityOperator(check_dtype=True)
      (<class 'pandas.core.frame.DataFrame'>): DataFrameEqualityOperator(nulls_compare_equal=False)
      (<class 'pandas.core.series.Series'>): SeriesEqualityOperator(nulls_compare_equal=False)
      (<class 'polars.dataframe.frame.DataFrame'>): DataFrameEqualityOperator()
      (<class 'polars.series.series.Series'>): SeriesEqualityOperator()
      (<class 'torch.Tensor'>): TensorEqualityOperator()
      (<class 'torch.nn.utils.rnn.PackedSequence'>): PackedSequenceEqualityOperator()
      (<class 'xarray.core.dataset.Dataset'>): DatasetEqualityOperator()
      (<class 'xarray.core.dataarray.DataArray'>): DataArrayEqualityOperator()
      (<class 'xarray.core.variable.Variable'>): VariableEqualityOperator()
    )

    ```
    """
    from coola.testers.equality import EqualityTester

    for typ, op in get_mapping_equality().items():
        if not EqualityTester.has_operator(typ):  # pragma: no cover
            EqualityTester.add_operator(typ, op)


register_allclose()
register_equality()

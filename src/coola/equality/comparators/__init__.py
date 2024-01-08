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
      (<class 'pandas...DataFrame'>): DataFrameEqualityOperator(nulls_compare_equal=False)
      (<class 'pandas...Series'>): SeriesEqualityOperator(nulls_compare_equal=False)
      (<class 'polars...DataFrame'>): DataFrameEqualityOperator()
      (<class 'polars.series.series.Series'>): SeriesEqualityOperator()
      (<class 'torch.Tensor'>): TensorEqualityOperator()
      (<class 'torch.nn.utils.rnn.PackedSequence'>): PackedSequenceEqualityOperator()
      (<class 'xarray.core.dataset.Dataset'>): DatasetEqualityOperator()
      (<class 'xarray.core.dataarray.DataArray'>): DataArrayEqualityOperator()
      (<class 'xarray.core.variable.Variable'>): VariableEqualityOperator()
    )

    ```
    """
    # Local import to avoid cyclic dependency
    from coola.equality.testers import EqualityTester

    for typ, op in get_type_comparator_mapping().items():
        if not EqualityTester.has_operator(typ):  # pragma: no cover
            EqualityTester.add_operator(typ, op)


register_equality()

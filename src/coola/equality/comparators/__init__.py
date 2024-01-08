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
    "register_equality",
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
    r"""Register equality comparators to ``EqualityTester``.

    >>> from coola.equality.comparators import register_equality
    >>> from coola.equality.testers import EqualityTester
    >>> register_equality()
    >>> tester = EqualityTester()
    >>> tester
    EqualityTester(
      (<class 'object'>): DefaultEqualityComparator()
      (<class 'collections.abc.Mapping'>): MappingEqualityComparator()
      (<class 'collections.abc.Sequence'>): SequenceEqualityComparator()
      (<class 'dict'>): MappingEqualityComparator()
      (<class 'list'>): SequenceEqualityComparator()
      (<class 'tuple'>): SequenceEqualityComparator()
      (<class 'jax.Array'>): JaxArrayEqualityComparator()
      (<class 'jaxlib.xla_extension.ArrayImpl'>): JaxArrayEqualityComparator()
      (<class 'numpy.ndarray'>): NumpyArrayEqualityComparator()
      (<class 'numpy.ma...MaskedArray'>): NumpyMaskedArrayEqualityComparator()
      (<class 'pandas...DataFrame'>): PandasDataFrameEqualityComparator()
      (<class 'pandas...Series'>): PandasSeriesEqualityComparator()
      (<class 'polars...DataFrame'>): PolarsDataFrameEqualityComparator()
      (<class 'polars...Series'>): PolarsSeriesEqualityComparator()
      (<class 'torch.nn.utils.rnn.PackedSequence'>): TorchPackedSequenceEqualityComparator()
      (<class 'torch.Tensor'>): TorchTensorEqualityComparator()
      (<class 'xarray...DataArray'>): XarrayDataArrayEqualityComparator()
      (<class 'xarray...Dataset'>): XarrayDatasetEqualityComparator()
      (<class 'xarray...Variable'>): XarrayVariableEqualityComparator()
    )

    ```
    """
    # Local import to avoid cyclic dependency
    from coola.equality.testers import EqualityTester

    for typ, op in get_type_comparator_mapping().items():
        if not EqualityTester.has_comparator(typ):  # pragma: no cover
            EqualityTester.add_comparator(typ, op)


register_equality()

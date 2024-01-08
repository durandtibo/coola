r"""Implement some utility functions for equality comparators."""

from __future__ import annotations

__all__ = ["get_type_comparator_mapping"]


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coola.equality.comparators.base import BaseEqualityComparator


def get_type_comparator_mapping() -> dict[type, BaseEqualityComparator]:
    r"""Get a default mapping between the types and the equality
    comparators.

    Returns:
        The mapping between the types and the equality comparators.

    ```pycon
    >>> from coola.equality.comparators import get_type_comparator_mapping
    >>> get_type_comparator_mapping()
    {<class 'object'>: DefaultEqualityComparator(),
     <class 'collections.abc.Mapping'>: MappingEqualityComparator(),
     <class 'collections.abc.Sequence'>: SequenceEqualityComparator(),
     <class 'dict'>: MappingEqualityComparator(),
     <class 'list'>: SequenceEqualityComparator(),
     <class 'tuple'>: SequenceEqualityComparator(),
     <class 'jax.Array'>: JaxArrayEqualityComparator(),
     <class 'jaxlib.xla_extension.ArrayImpl'>: JaxArrayEqualityComparator(),
     <class 'numpy.ndarray'>: NumpyArrayEqualityComparator(),
     <class 'numpy.ma...MaskedArray'>: NumpyMaskedArrayEqualityComparator(),
     <class 'pandas...DataFrame'>: PandasDataFrameEqualityComparator(),
     <class 'pandas...Series'>: PandasSeriesEqualityComparator(),
     <class 'polars...DataFrame'>: PolarsDataFrameEqualityComparator(),
     <class 'polars...Series'>: PolarsSeriesEqualityComparator(),
     <class 'torch.nn.utils.rnn.PackedSequence'>: TorchPackedSequenceEqualityComparator(),
     <class 'torch.Tensor'>: TorchTensorEqualityComparator(),
     <class 'xarray...DataArray'>: XarrayDataArrayEqualityComparator(),
     <class 'xarray...Dataset'>: XarrayDatasetEqualityComparator(),
     <class 'xarray...Variable'>: XarrayVariableEqualityComparator()}

    ```
    """
    from coola.equality import comparators  # Local import to avoid cyclic dependencies

    return (
        comparators.default.get_type_comparator_mapping()
        | comparators.collection.get_type_comparator_mapping()
        | comparators.jax_.get_type_comparator_mapping()
        | comparators.numpy_.get_type_comparator_mapping()
        | comparators.pandas_.get_type_comparator_mapping()
        | comparators.polars_.get_type_comparator_mapping()
        | comparators.torch_.get_type_comparator_mapping()
        | comparators.xarray_.get_type_comparator_mapping()
    )

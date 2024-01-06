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
    # >>> from coola.equality.comparators import get_type_comparator_mapping
    # >>> get_type_comparator_mapping()
    # {<class 'collections.abc.Mapping'>: MappingEqualityOperator(),
    #  <class 'collections.abc.Sequence'>: SequenceEqualityOperator(),
    #  <class 'dict'>: MappingEqualityOperator(),
    #  <class 'list'>: SequenceEqualityOperator(),
    #  <class 'object'>: DefaultEqualityOperator(),
    #  <class 'tuple'>: SequenceEqualityOperator(),
    #  <class 'jax.Array'>: JaxArrayEqualityOperator(check_dtype=True),
    #  <class 'jaxlib.xla_extension.ArrayImpl'>: JaxArrayEqualityOperator(check_dtype=True),
    #  <class 'numpy.ndarray'>: ArrayEqualityOperator(check_dtype=True),
    #  <class 'pandas.core.frame.DataFrame'>: DataFrameEqualityOperator(nulls_compare_equal=False),
    #  <class 'pandas.core.series.Series'>: SeriesEqualityOperator(nulls_compare_equal=False),
    #  <class 'polars.dataframe.frame.DataFrame'>: DataFrameEqualityOperator(),
    #  <class 'polars.series.series.Series'>: SeriesEqualityOperator(),
    #  <class 'torch.Tensor'>: TensorEqualityOperator(),
    #  <class 'torch.nn.utils.rnn.PackedSequence'>: PackedSequenceEqualityOperator(),
    #  <class 'xarray.core.dataset.Dataset'>: DatasetEqualityOperator(),
    #  <class 'xarray.core.dataarray.DataArray'>: DataArrayEqualityOperator(),
    #  <class 'xarray.core.variable.Variable'>: VariableEqualityOperator()}

    ```
    """
    from coola.equality import comparators  # Local import to avoid cyclic dependencies

    return (
        comparators.default.get_type_comparator_mapping()
        # | comparators.jax_.get_type_comparator_mapping()
        | comparators.numpy_.get_type_comparator_mapping()
        # | comparators.pandas_.get_type_comparator_mapping()
        # | comparators.polars_.get_type_comparator_mapping()
        # | comparators.torch_.get_type_comparator_mapping()
        # | comparators.xarray_.get_type_comparator_mapping()
    )

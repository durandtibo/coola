from __future__ import annotations

__all__ = ["get_mapping_allclose", "get_mapping_equality"]


from coola.comparators.base import BaseAllCloseOperator, BaseEqualityOperator


def get_mapping_allclose() -> dict[type[object], BaseAllCloseOperator]:
    r"""Gets a default mapping between the types and the allclose
    operators.

    Returns:
        dict: The mapping between the types and the allclose
            operators.

    ```pycon
    >>> from coola.comparators import get_mapping_allclose
    >>> get_mapping_allclose()
    {<class 'collections.abc.Mapping'>: MappingAllCloseOperator(),
     <class 'collections.abc.Sequence'>: SequenceAllCloseOperator(),
     <class 'bool'>: ScalarAllCloseOperator(),
     <class 'dict'>: MappingAllCloseOperator(),
     <class 'float'>: ScalarAllCloseOperator(),
     <class 'int'>: ScalarAllCloseOperator(),
     <class 'list'>: SequenceAllCloseOperator(),
     <class 'object'>: DefaultAllCloseOperator(),
     <class 'tuple'>: SequenceAllCloseOperator(),
     <class 'jax.Array'>: JaxArrayAllCloseOperator(check_dtype=True),
     <class 'jaxlib.xla_extension.ArrayImpl'>: JaxArrayAllCloseOperator(check_dtype=True),
     <class 'numpy.ndarray'>: ArrayAllCloseOperator(check_dtype=True),
     <class 'pandas.core.frame.DataFrame'>: DataFrameAllCloseOperator(),
     <class 'pandas.core.series.Series'>: SeriesAllCloseOperator(),
     <class 'polars.dataframe.frame.DataFrame'>: DataFrameAllCloseOperator(),
     <class 'polars.series.series.Series'>: SeriesAllCloseOperator(),
     <class 'torch.Tensor'>: TensorAllCloseOperator(),
     <class 'torch.nn.utils.rnn.PackedSequence'>: PackedSequenceAllCloseOperator(),
     <class 'xarray.core.dataset.Dataset'>: DatasetAllCloseOperator(),
     <class 'xarray.core.dataarray.DataArray'>: DataArrayAllCloseOperator(),
     <class 'xarray.core.variable.Variable'>: VariableAllCloseOperator()}

    ```
    """
    from coola import comparators  # Local import to avoid cyclic dependencies

    return (
        comparators.allclose.get_mapping_allclose()
        | comparators.jax_.get_mapping_allclose()
        | comparators.numpy_.get_mapping_allclose()
        | comparators.pandas_.get_mapping_allclose()
        | comparators.polars_.get_mapping_allclose()
        | comparators.torch_.get_mapping_allclose()
        | comparators.xarray_.get_mapping_allclose()
    )


def get_mapping_equality() -> dict[type[object], BaseEqualityOperator]:
    r"""Gets a default mapping between the types and the equality
    operators.

    Returns:
        dict: The mapping between the types and the equality
            operators.

    ```pycon
    >>> from coola.comparators import get_mapping_equality
    >>> get_mapping_equality()
    {<class 'collections.abc.Mapping'>: MappingEqualityOperator(),
     <class 'collections.abc.Sequence'>: SequenceEqualityOperator(),
     <class 'dict'>: MappingEqualityOperator(),
     <class 'list'>: SequenceEqualityOperator(),
     <class 'object'>: DefaultEqualityOperator(),
     <class 'tuple'>: SequenceEqualityOperator(),
     <class 'jax.Array'>: JaxArrayEqualityOperator(check_dtype=True),
     <class 'jaxlib.xla_extension.ArrayImpl'>: JaxArrayEqualityOperator(check_dtype=True),
     <class 'numpy.ndarray'>: ArrayEqualityOperator(check_dtype=True),
     <class 'pandas.core.frame.DataFrame'>: DataFrameEqualityOperator(nulls_compare_equal=False),
     <class 'pandas.core.series.Series'>: SeriesEqualityOperator(nulls_compare_equal=False),
     <class 'polars.dataframe.frame.DataFrame'>: DataFrameEqualityOperator(),
     <class 'polars.series.series.Series'>: SeriesEqualityOperator(),
     <class 'torch.Tensor'>: TensorEqualityOperator(),
     <class 'torch.nn.utils.rnn.PackedSequence'>: PackedSequenceEqualityOperator(),
     <class 'xarray.core.dataset.Dataset'>: DatasetEqualityOperator(),
     <class 'xarray.core.dataarray.DataArray'>: DataArrayEqualityOperator(),
     <class 'xarray.core.variable.Variable'>: VariableEqualityOperator()}

    ```
    """
    from coola import comparators  # Local import to avoid cyclic dependencies

    return (
        comparators.equality.get_mapping_equality()
        | comparators.jax_.get_mapping_equality()
        | comparators.numpy_.get_mapping_equality()
        | comparators.pandas_.get_mapping_equality()
        | comparators.polars_.get_mapping_equality()
        | comparators.torch_.get_mapping_equality()
        | comparators.xarray_.get_mapping_equality()
    )

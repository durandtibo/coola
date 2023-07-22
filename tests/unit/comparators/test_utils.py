from collections.abc import Mapping, Sequence

from coola.comparators import (
    DataArrayAllCloseOperator,
    DataArrayEqualityOperator,
    DatasetAllCloseOperator,
    DatasetEqualityOperator,
    DefaultAllCloseOperator,
    DefaultEqualityOperator,
    MappingAllCloseOperator,
    MappingEqualityOperator,
    NDArrayAllCloseOperator,
    NDArrayEqualityOperator,
    PackedSequenceAllCloseOperator,
    PackedSequenceEqualityOperator,
    ScalarAllCloseOperator,
    SequenceAllCloseOperator,
    SequenceEqualityOperator,
    TensorAllCloseOperator,
    TensorEqualityOperator,
    VariableAllCloseOperator,
    VariableEqualityOperator,
    get_mapping_allclose,
    get_mapping_equality,
)
from coola.testing import (
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
    xarray_available,
)
from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available,
)

if is_numpy_available():
    import numpy as np

if is_pandas_available():
    import pandas

if is_polars_available():
    import polars

if is_torch_available():
    import torch

if is_xarray_available():
    import xarray as xr


##########################################
#     Tests for get_mapping_allclose     #
##########################################


def test_get_mapping_allclose() -> None:
    mapping = get_mapping_allclose()
    assert len(mapping) >= 9
    assert isinstance(mapping[Mapping], MappingAllCloseOperator)
    assert isinstance(mapping[Sequence], SequenceAllCloseOperator)
    assert isinstance(mapping[bool], ScalarAllCloseOperator)
    assert isinstance(mapping[dict], MappingAllCloseOperator)
    assert isinstance(mapping[float], ScalarAllCloseOperator)
    assert isinstance(mapping[int], ScalarAllCloseOperator)
    assert isinstance(mapping[list], SequenceAllCloseOperator)
    assert isinstance(mapping[object], DefaultAllCloseOperator)
    assert isinstance(mapping[tuple], SequenceAllCloseOperator)


@numpy_available
def test_get_mapping_allclose_numpy() -> None:
    mapping = get_mapping_allclose()
    assert isinstance(mapping[np.ndarray], NDArrayAllCloseOperator)


@pandas_available
def test_get_mapping_allclose_pandas() -> None:
    from coola.comparators.pandas_ import (
        DataFrameAllCloseOperator,
        SeriesAllCloseOperator,
    )

    mapping = get_mapping_allclose()
    assert isinstance(mapping[pandas.DataFrame], DataFrameAllCloseOperator)
    assert isinstance(mapping[pandas.Series], SeriesAllCloseOperator)


@polars_available
def test_get_mapping_allclose_polars() -> None:
    from coola.comparators.polars_ import (
        DataFrameAllCloseOperator,
        SeriesAllCloseOperator,
    )

    mapping = get_mapping_allclose()
    assert isinstance(mapping[polars.DataFrame], DataFrameAllCloseOperator)
    assert isinstance(mapping[polars.Series], SeriesAllCloseOperator)


@torch_available
def test_get_mapping_allclose_torch() -> None:
    mapping = get_mapping_allclose()
    assert isinstance(mapping[torch.Tensor], TensorAllCloseOperator)
    assert isinstance(mapping[torch.nn.utils.rnn.PackedSequence], PackedSequenceAllCloseOperator)


@xarray_available
def test_get_mapping_allclose_xarray() -> None:
    mapping = get_mapping_allclose()
    assert isinstance(mapping[xr.DataArray], DataArrayAllCloseOperator)
    assert isinstance(mapping[xr.Dataset], DatasetAllCloseOperator)
    assert isinstance(mapping[xr.Variable], VariableAllCloseOperator)


##########################################
#     Tests for get_mapping_equality     #
##########################################


def test_get_mapping_equality() -> None:
    mapping = get_mapping_equality()
    assert len(mapping) >= 9
    assert isinstance(mapping[Mapping], MappingEqualityOperator)
    assert isinstance(mapping[Sequence], SequenceEqualityOperator)
    assert isinstance(mapping[dict], MappingEqualityOperator)
    assert isinstance(mapping[list], SequenceEqualityOperator)
    assert isinstance(mapping[object], DefaultEqualityOperator)
    assert isinstance(mapping[tuple], SequenceEqualityOperator)


@numpy_available
def test_get_mapping_equality_numpy() -> None:
    mapping = get_mapping_equality()
    assert isinstance(mapping[np.ndarray], NDArrayEqualityOperator)


@pandas_available
def test_get_mapping_equality_pandas() -> None:
    from coola.comparators.pandas_ import (
        DataFrameEqualityOperator,
        SeriesEqualityOperator,
    )

    mapping = get_mapping_equality()
    assert isinstance(mapping[pandas.DataFrame], DataFrameEqualityOperator)
    assert isinstance(mapping[pandas.Series], SeriesEqualityOperator)


@polars_available
def test_get_mapping_equality_polars() -> None:
    from coola.comparators.polars_ import (
        DataFrameEqualityOperator,
        SeriesEqualityOperator,
    )

    mapping = get_mapping_equality()
    assert isinstance(mapping[polars.DataFrame], DataFrameEqualityOperator)
    assert isinstance(mapping[polars.Series], SeriesEqualityOperator)


@torch_available
def test_get_mapping_equality_torch() -> None:
    mapping = get_mapping_equality()
    assert isinstance(mapping[torch.Tensor], TensorEqualityOperator)
    assert isinstance(mapping[torch.nn.utils.rnn.PackedSequence], PackedSequenceEqualityOperator)


@xarray_available
def test_get_mapping_equality_xarray() -> None:
    mapping = get_mapping_equality()
    assert isinstance(mapping[xr.DataArray], DataArrayEqualityOperator)
    assert isinstance(mapping[xr.Dataset], DatasetEqualityOperator)
    assert isinstance(mapping[xr.Variable], VariableEqualityOperator)

from __future__ import annotations

from collections.abc import Mapping, Sequence

from coola.equality.comparators import (
    DefaultEqualityComparator,
    JaxArrayEqualityComparator,
    MappingEqualityComparator,
    NumpyArrayEqualityComparator,
    NumpyMaskedArrayEqualityComparator,
    PandasDataFrameEqualityComparator,
    PandasSeriesEqualityComparator,
    PolarsDataFrameEqualityComparator,
    PolarsSeriesEqualityComparator,
    SequenceEqualityComparator,
    TorchPackedSequenceEqualityComparator,
    TorchTensorEqualityComparator,
    XarrayDataArrayEqualityComparator,
    XarrayDatasetEqualityComparator,
    XarrayVariableEqualityComparator,
    get_type_comparator_mapping,
)
from coola.testing import (
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
    xarray_available,
)
from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available,
)

if is_jax_available():
    import jax.numpy as jnp
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

#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    mapping = get_type_comparator_mapping()
    assert len(mapping) >= 6
    assert isinstance(mapping[Mapping], MappingEqualityComparator)
    assert isinstance(mapping[Sequence], SequenceEqualityComparator)
    assert isinstance(mapping[dict], MappingEqualityComparator)
    assert isinstance(mapping[list], SequenceEqualityComparator)
    assert isinstance(mapping[object], DefaultEqualityComparator)
    assert isinstance(mapping[tuple], SequenceEqualityComparator)


@jax_available
def test_get_type_comparator_mapping_jax() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[jnp.ndarray], JaxArrayEqualityComparator)


@numpy_available
def test_get_type_comparator_mapping_numpy() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[np.ndarray], NumpyArrayEqualityComparator)
    assert isinstance(mapping[np.ma.MaskedArray], NumpyMaskedArrayEqualityComparator)


@pandas_available
def test_get_type_comparator_mapping_pandas() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[pandas.DataFrame], PandasDataFrameEqualityComparator)
    assert isinstance(mapping[pandas.Series], PandasSeriesEqualityComparator)


@polars_available
def test_get_type_comparator_mapping_polars() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[polars.DataFrame], PolarsDataFrameEqualityComparator)
    assert isinstance(mapping[polars.Series], PolarsSeriesEqualityComparator)


@torch_available
def test_get_type_comparator_mapping_torch() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(
        mapping[torch.nn.utils.rnn.PackedSequence], TorchPackedSequenceEqualityComparator
    )
    assert isinstance(mapping[torch.Tensor], TorchTensorEqualityComparator)


@xarray_available
def test_get_type_comparator_mapping_xarray() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[xr.DataArray], XarrayDataArrayEqualityComparator)
    assert isinstance(mapping[xr.Dataset], XarrayDatasetEqualityComparator)
    assert isinstance(mapping[xr.Variable], XarrayVariableEqualityComparator)

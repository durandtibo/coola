from __future__ import annotations

import pytest

from coola.hashing import hash_object
from coola.testing.fixtures import (
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    torch_available,
    xarray_available,
)
from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
    is_xarray_available,
)

if is_jax_available():
    import jax.numpy as jnp
if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_pyarrow_available():
    import pyarrow as pa
if is_torch_available():
    import torch
if is_xarray_available():
    import xarray as xr


@numpy_available
def test_hash_numpy_array_equal_arrays() -> None:
    assert hash_object(np.arange(6).reshape(2, 3)) == hash_object(np.arange(6).reshape(2, 3))


@numpy_available
def test_hash_numpy_array_length() -> None:
    assert len(hash_object(np.arange(3), length=16)) == 16


@numpy_available
@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(
            np.arange(3) if is_numpy_available() else 0,
            np.arange(3.0) if is_numpy_available() else 0,
            id="dtype",
        ),
        pytest.param(
            np.arange(6) if is_numpy_available() else 0,
            np.arange(6).reshape(2, 3) if is_numpy_available() else 0,
            id="shape",
        ),
        pytest.param(
            np.arange(3) if is_numpy_available() else 0,
            np.arange(1, 4) if is_numpy_available() else 0,
            id="values",
        ),
    ],
)
def test_hash_numpy_array_different(first: object, second: object) -> None:
    assert hash_object(first) != hash_object(second)


@numpy_available
def test_hash_numpy_array_ignores_memory_layout() -> None:
    array = np.arange(6).reshape(2, 3)
    assert hash_object(array.T) == hash_object(np.ascontiguousarray(array.T))


@numpy_available
def test_hash_numpy_array_negative_zero() -> None:
    assert hash_object(np.array([0.0])) == hash_object(np.array([-0.0]))


@numpy_available
def test_hash_numpy_array_nan_sign() -> None:
    assert hash_object(np.array([np.nan])) == hash_object(np.array([-np.nan]))


@numpy_available
def test_hash_numpy_array_nan_payload() -> None:
    payload = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    assert hash_object(np.array([np.nan])) == hash_object(payload)


@numpy_available
def test_hash_numpy_array_float32_nan_sign() -> None:
    first, second = np.array([np.nan], dtype=np.float32), np.array([-np.nan], dtype=np.float32)
    assert hash_object(first) == hash_object(second)


@numpy_available
def test_hash_numpy_array_nan_differs_from_a_number() -> None:
    assert hash_object(np.array([np.nan])) != hash_object(np.array([0.0]))


@numpy_available
def test_hash_numpy_array_nan_differs_from_inf() -> None:
    assert hash_object(np.array([np.nan])) != hash_object(np.array([np.inf]))


@numpy_available
def test_hash_numpy_array_complex_nan_sign() -> None:
    first, second = np.array([complex(np.nan, 1.0)]), np.array([complex(-np.nan, 1.0)])
    assert hash_object(first) == hash_object(second)


@numpy_available
def test_hash_numpy_array_complex_nan_keeps_the_other_part() -> None:
    first, second = np.array([complex(np.nan, 1.0)]), np.array([complex(np.nan, 2.0)])
    assert hash_object(first) != hash_object(second)


@numpy_available
def test_hash_numpy_array_nan_does_not_modify_the_input() -> None:
    array = np.array([-np.nan])
    hash_object(array)
    assert np.signbit(array[0])


@numpy_available
def test_hash_numpy_array_zero_dimension_nan_sign() -> None:
    assert hash_object(np.array(np.nan)) == hash_object(np.array(-np.nan))


@numpy_available
def test_hash_numpy_array_object_dtype_equal_arrays() -> None:
    first = np.array(["a", None, 1], dtype=object)
    assert hash_object(first) == hash_object(np.array(["a", None, 1], dtype=object))


@numpy_available
def test_hash_numpy_array_object_dtype_different_values() -> None:
    first = np.array(["a", None, 1], dtype=object)
    assert hash_object(first) != hash_object(np.array(["a", None, 2], dtype=object))


@numpy_available
def test_hash_numpy_masked_array_ignores_masked_values() -> None:
    first = np.ma.array([1, 2], mask=[0, 1])
    assert hash_object(first) == hash_object(np.ma.array([1, 3], mask=[0, 1]))


@numpy_available
def test_hash_numpy_masked_array_differs_with_another_mask() -> None:
    first = np.ma.array([1, 2], mask=[0, 1])
    assert hash_object(first) != hash_object(np.ma.array([1, 2], mask=[0, 0]))


@numpy_available
def test_hash_numpy_masked_array_differs_from_unmasked_array() -> None:
    first = np.ma.array([1, 2], mask=[0, 1])
    assert hash_object(first) != hash_object(np.array([1, 2]))


@numpy_available
def test_hash_numpy_array_in_container() -> None:
    assert hash_object({"a": [np.arange(3)]}) == hash_object({"a": [np.arange(3)]})


@torch_available
def test_hash_torch_tensor_equal_tensors() -> None:
    assert hash_object(torch.arange(4.0)) == hash_object(torch.arange(4.0))


@torch_available
def test_hash_torch_tensor_differs_by_dtype() -> None:
    assert hash_object(torch.arange(4.0)) != hash_object(torch.arange(4))


@torch_available
def test_hash_torch_tensor_differs_by_shape() -> None:
    assert hash_object(torch.zeros(2, 3)) != hash_object(torch.zeros(3, 2))


@torch_available
def test_hash_torch_tensor_negative_zero() -> None:
    assert hash_object(torch.tensor([0.0])) == hash_object(torch.tensor([-0.0]))


@torch_available
def test_hash_torch_tensor_nan_sign() -> None:
    nan = torch.tensor([float("nan")])
    assert hash_object(nan) == hash_object(-nan)


@torch_available
def test_hash_torch_tensor_nan_differs_from_a_number() -> None:
    assert hash_object(torch.tensor([float("nan")])) != hash_object(torch.tensor([0.0]))


@torch_available
def test_hash_torch_tensor_complex_nan_sign() -> None:
    first = torch.tensor([complex(float("nan"), 1.0)])
    second = torch.tensor([complex(-float("nan"), 1.0)])
    assert hash_object(first) == hash_object(second)


@torch_available
def test_hash_torch_tensor_complex_nan_keeps_the_other_part() -> None:
    first = torch.tensor([complex(float("nan"), 1.0)])
    second = torch.tensor([complex(float("nan"), 2.0)])
    assert hash_object(first) != hash_object(second)


@torch_available
def test_hash_torch_tensor_bfloat16_nan_sign() -> None:
    nan = torch.tensor([float("nan")]).bfloat16()
    assert hash_object(nan) == hash_object(-nan)


@torch_available
@pytest.mark.parametrize(
    "tensor",
    [
        pytest.param(lambda: torch.tensor([True, False]), id="bool"),
        pytest.param(lambda: torch.tensor([1 + 2j]), id="complex"),
        pytest.param(lambda: torch.arange(4.0, dtype=torch.bfloat16), id="bfloat16"),
        pytest.param(lambda: torch.tensor(1.0), id="scalar"),
        pytest.param(lambda: torch.arange(6.0).reshape(2, 3).T, id="non contiguous"),
    ],
)
def test_hash_torch_tensor_dtypes(tensor: object) -> None:
    assert hash_object(tensor()) == hash_object(tensor())


@torch_available
def test_hash_torch_sparse_tensor() -> None:
    assert hash_object(torch.eye(2).to_sparse()) == hash_object(torch.eye(2).to_sparse())


@torch_available
@numpy_available
def test_hash_torch_tensor_differs_from_numpy_array() -> None:
    assert hash_object(torch.arange(3)) != hash_object(np.arange(3))


@jax_available
def test_hash_jax_array_equal_arrays() -> None:
    assert hash_object(jnp.arange(3)) == hash_object(jnp.arange(3))


@jax_available
def test_hash_jax_array_differs_by_dtype() -> None:
    assert hash_object(jnp.arange(3)) != hash_object(jnp.arange(3.0))


@jax_available
def test_hash_jax_array_negative_zero() -> None:
    assert hash_object(jnp.array([0.0])) == hash_object(jnp.array([-0.0]))


@jax_available
def test_hash_jax_array_nan_sign() -> None:
    nan = jnp.array([jnp.nan])
    assert hash_object(nan) == hash_object(-nan)


@pandas_available
def test_hash_pandas_dataframe_equal_frames() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(frame) == hash_object(frame.copy())


@pandas_available
def test_hash_pandas_dataframe_differs_by_value() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(frame) != hash_object(frame.assign(a=[1, 3]))


@pandas_available
def test_hash_pandas_dataframe_differs_by_column_name() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(frame) != hash_object(frame.rename(columns={"a": "c"}))


@pandas_available
def test_hash_pandas_dataframe_differs_by_index() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(frame) != hash_object(frame.set_axis([5, 6]))


@pandas_available
def test_hash_pandas_series_equal_series() -> None:
    series = pd.Series([1, 2], name="a")
    assert hash_object(series) == hash_object(series.copy())


@pandas_available
def test_hash_pandas_series_differs_by_name() -> None:
    series = pd.Series([1, 2], name="a")
    assert hash_object(series) != hash_object(series.rename("b"))


@pandas_available
def test_hash_pandas_series_differs_by_dtype() -> None:
    series = pd.Series([1, 2], name="a")
    assert hash_object(series) != hash_object(series.astype(float))


@pandas_available
def test_hash_pandas_series_nan_sign() -> None:
    assert hash_object(pd.Series([np.nan])) == hash_object(pd.Series([-np.nan]))


@polars_available
def test_hash_polars_dataframe_equal_frames() -> None:
    frame = pl.DataFrame({"a": [1, None], "b": ["x", "y"]})
    assert hash_object(frame) == hash_object(frame.clone())


@polars_available
def test_hash_polars_dataframe_differs_by_value() -> None:
    frame = pl.DataFrame({"a": [1, None], "b": ["x", "y"]})
    assert hash_object(frame) != hash_object(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


@polars_available
def test_hash_polars_dataframe_differs_by_column_name() -> None:
    frame = pl.DataFrame({"a": [1, None], "b": ["x", "y"]})
    assert hash_object(frame) != hash_object(frame.rename({"a": "c"}))


@polars_available
def test_hash_polars_series_equal_series() -> None:
    assert hash_object(pl.Series("a", [1.5])) == hash_object(pl.Series("a", [1.5]))


@polars_available
def test_hash_polars_series_differs_by_name() -> None:
    assert hash_object(pl.Series("a", [1.5])) != hash_object(pl.Series("b", [1.5]))


@pyarrow_available
def test_hash_pyarrow_array_equal_arrays() -> None:
    assert hash_object(pa.array([1, None])) == hash_object(pa.array([1, None]))


@pyarrow_available
def test_hash_pyarrow_array_differs_by_value() -> None:
    assert hash_object(pa.array([1, 2])) != hash_object(pa.array([1, 3]))


@pyarrow_available
def test_hash_pyarrow_array_differs_by_type() -> None:
    assert hash_object(pa.array([1, 2])) != hash_object(pa.array([1, 2], type=pa.int32()))


@pyarrow_available
def test_hash_pyarrow_array_chunked_array_equals_the_flat_array_layout() -> None:
    assert hash_object(pa.chunked_array([[1], [2]])) == hash_object(pa.chunked_array([[1, 2]]))


@pyarrow_available
def test_hash_pyarrow_table_equal_tables() -> None:
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(table) == hash_object(pa.table({"a": [1, 2], "b": ["x", "y"]}))


@pyarrow_available
def test_hash_pyarrow_table_differs_by_value() -> None:
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(table) != hash_object(pa.table({"a": [1, 3], "b": ["x", "y"]}))


@pyarrow_available
def test_hash_pyarrow_table_differs_by_column_name() -> None:
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(table) != hash_object(pa.table({"c": [1, 2], "b": ["x", "y"]}))


@pyarrow_available
def test_hash_pyarrow_table_record_batch_is_stable() -> None:
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(table.to_batches()[0]) == hash_object(table.to_batches()[0])


@xarray_available
def test_hash_xarray_data_array_equal() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]}, name="n")
    assert hash_object(array) == hash_object(array.copy())


@xarray_available
def test_hash_xarray_data_array_differs_by_name() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]}, name="n")
    assert hash_object(array) != hash_object(array.rename("m"))


@xarray_available
def test_hash_xarray_data_array_differs_by_coordinates() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]}, name="n")
    assert hash_object(array) != hash_object(array.assign_coords(x=[0, 2]))


@xarray_available
def test_hash_xarray_dataset_equal() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]}, name="n")
    assert hash_object(array.to_dataset()) == hash_object(array.copy().to_dataset())


@xarray_available
def test_hash_xarray_dataset_differs_by_name() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]}, name="n")
    assert hash_object(array.to_dataset()) != hash_object(array.rename("m").to_dataset())


@xarray_available
def test_hash_xarray_variable_equal() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]}, name="n")
    assert hash_object(array.variable) == hash_object(array.copy().variable)


@xarray_available
def test_hash_xarray_variable_differs_by_data() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]}, name="n")
    assert hash_object(array.variable) != hash_object(array.variable.copy(data=[1, 3]))

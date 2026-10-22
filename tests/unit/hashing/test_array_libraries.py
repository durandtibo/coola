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
def test_hash_numpy_array() -> None:
    assert hash_object(np.arange(6).reshape(2, 3)) == hash_object(np.arange(6).reshape(2, 3))
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
def test_hash_numpy_array_object_dtype() -> None:
    first = np.array(["a", None, 1], dtype=object)
    assert hash_object(first) == hash_object(np.array(["a", None, 1], dtype=object))
    assert hash_object(first) != hash_object(np.array(["a", None, 2], dtype=object))


@numpy_available
def test_hash_numpy_masked_array() -> None:
    first = np.ma.array([1, 2], mask=[0, 1])
    assert hash_object(first) == hash_object(np.ma.array([1, 3], mask=[0, 1]))
    assert hash_object(first) != hash_object(np.ma.array([1, 2], mask=[0, 0]))
    assert hash_object(first) != hash_object(np.array([1, 2]))


@numpy_available
def test_hash_numpy_array_in_container() -> None:
    assert hash_object({"a": [np.arange(3)]}) == hash_object({"a": [np.arange(3)]})


@torch_available
def test_hash_torch_tensor() -> None:
    assert hash_object(torch.arange(4.0)) == hash_object(torch.arange(4.0))
    assert hash_object(torch.arange(4.0)) != hash_object(torch.arange(4))
    assert hash_object(torch.zeros(2, 3)) != hash_object(torch.zeros(3, 2))
    assert hash_object(torch.tensor([0.0])) == hash_object(torch.tensor([-0.0]))


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
def test_hash_jax_array() -> None:
    assert hash_object(jnp.arange(3)) == hash_object(jnp.arange(3))
    assert hash_object(jnp.arange(3)) != hash_object(jnp.arange(3.0))
    assert hash_object(jnp.array([0.0])) == hash_object(jnp.array([-0.0]))


@pandas_available
def test_hash_pandas_dataframe() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(frame) == hash_object(frame.copy())
    assert hash_object(frame) != hash_object(frame.assign(a=[1, 3]))
    assert hash_object(frame) != hash_object(frame.rename(columns={"a": "c"}))
    assert hash_object(frame) != hash_object(frame.set_axis([5, 6]))


@pandas_available
def test_hash_pandas_series() -> None:
    series = pd.Series([1, 2], name="a")
    assert hash_object(series) == hash_object(series.copy())
    assert hash_object(series) != hash_object(series.rename("b"))
    assert hash_object(series) != hash_object(series.astype(float))


@polars_available
def test_hash_polars_dataframe() -> None:
    frame = pl.DataFrame({"a": [1, None], "b": ["x", "y"]})
    assert hash_object(frame) == hash_object(frame.clone())
    assert hash_object(frame) != hash_object(pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
    assert hash_object(frame) != hash_object(frame.rename({"a": "c"}))


@polars_available
def test_hash_polars_series() -> None:
    assert hash_object(pl.Series("a", [1.5])) == hash_object(pl.Series("a", [1.5]))
    assert hash_object(pl.Series("a", [1.5])) != hash_object(pl.Series("b", [1.5]))


@pyarrow_available
def test_hash_pyarrow_array() -> None:
    assert hash_object(pa.array([1, None])) == hash_object(pa.array([1, None]))
    assert hash_object(pa.array([1, 2])) != hash_object(pa.array([1, 3]))
    assert hash_object(pa.array([1, 2])) != hash_object(pa.array([1, 2], type=pa.int32()))
    assert hash_object(pa.chunked_array([[1], [2]])) == hash_object(pa.chunked_array([[1, 2]]))


@pyarrow_available
def test_hash_pyarrow_table() -> None:
    table = pa.table({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object(table) == hash_object(pa.table({"a": [1, 2], "b": ["x", "y"]}))
    assert hash_object(table) != hash_object(pa.table({"a": [1, 3], "b": ["x", "y"]}))
    assert hash_object(table) != hash_object(pa.table({"c": [1, 2], "b": ["x", "y"]}))
    assert hash_object(table.to_batches()[0]) == hash_object(table.to_batches()[0])


@xarray_available
def test_hash_xarray() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]}, name="n")
    assert hash_object(array) == hash_object(array.copy())
    assert hash_object(array) != hash_object(array.rename("m"))
    assert hash_object(array) != hash_object(array.assign_coords(x=[0, 2]))
    assert hash_object(array.to_dataset()) == hash_object(array.copy().to_dataset())
    assert hash_object(array.to_dataset()) != hash_object(array.rename("m").to_dataset())
    assert hash_object(array.variable) == hash_object(array.copy().variable)
    assert hash_object(array.variable) != hash_object(array.variable.copy(data=[1, 3]))

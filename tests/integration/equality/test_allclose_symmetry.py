from __future__ import annotations

from typing import Any

import pytest

from coola.equality import objects_are_allclose
from coola.testing.marks import (
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

if is_numpy_available():
    import numpy as np
if is_torch_available():
    import torch
if is_jax_available():
    import jax.numpy as jnp
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_pyarrow_available():
    import pyarrow as pa
if is_xarray_available():
    import xarray as xr

KWARGS: dict[str, Any] = {"rtol": 0.01, "atol": 0.0}


def test_objects_are_allclose_scalar_is_symmetric() -> None:
    assert objects_are_allclose(100.0, 101.005, **KWARGS) == objects_are_allclose(
        101.005, 100.0, **KWARGS
    )


@numpy_available
@pytest.mark.parametrize(("actual", "expected"), [(100.0, 101.005), (101.005, 100.0)])
def test_objects_are_allclose_numpy_is_symmetric(actual: float, expected: float) -> None:
    assert objects_are_allclose(np.array([actual]), np.array([expected]), **KWARGS)


@torch_available
@pytest.mark.parametrize(("actual", "expected"), [(100.0, 101.005), (101.005, 100.0)])
def test_objects_are_allclose_torch_is_symmetric(actual: float, expected: float) -> None:
    assert objects_are_allclose(torch.tensor([actual]), torch.tensor([expected]), **KWARGS)


@jax_available
@pytest.mark.parametrize(("actual", "expected"), [(100.0, 101.005), (101.005, 100.0)])
def test_objects_are_allclose_jax_is_symmetric(actual: float, expected: float) -> None:
    assert objects_are_allclose(jnp.array([actual]), jnp.array([expected]), **KWARGS)


@numpy_available
def test_objects_are_allclose_numpy_nested_in_a_dict_is_symmetric() -> None:
    assert objects_are_allclose({"a": np.array([101.005])}, {"a": np.array([100.0])}, **KWARGS)


################################################
#     Tests for the agreement of containers    #
################################################

# Each element needs a different direction: the answer must not depend on the container.
OPPOSITE_X = [100.0, 101.005]
OPPOSITE_Y = [101.005, 100.0]


def test_objects_are_allclose_list_opposite_directions() -> None:
    assert objects_are_allclose(OPPOSITE_X, OPPOSITE_Y, **KWARGS)


@numpy_available
def test_objects_are_allclose_numpy_opposite_directions() -> None:
    assert objects_are_allclose(np.array(OPPOSITE_X), np.array(OPPOSITE_Y), **KWARGS)


@torch_available
def test_objects_are_allclose_torch_opposite_directions() -> None:
    assert objects_are_allclose(
        torch.tensor(OPPOSITE_X, dtype=torch.float64),
        torch.tensor(OPPOSITE_Y, dtype=torch.float64),
        **KWARGS,
    )


@jax_available
def test_objects_are_allclose_jax_opposite_directions() -> None:
    assert objects_are_allclose(jnp.array(OPPOSITE_X), jnp.array(OPPOSITE_Y), **KWARGS)


@pyarrow_available
def test_objects_are_allclose_pyarrow_opposite_directions() -> None:
    assert objects_are_allclose(pa.array(OPPOSITE_X), pa.array(OPPOSITE_Y), **KWARGS)


@xarray_available
def test_objects_are_allclose_xarray_opposite_directions() -> None:
    assert objects_are_allclose(xr.DataArray(OPPOSITE_X), xr.DataArray(OPPOSITE_Y), **KWARGS)


@pandas_available
def test_objects_are_allclose_pandas_opposite_directions() -> None:
    assert objects_are_allclose(pd.Series(OPPOSITE_X), pd.Series(OPPOSITE_Y), **KWARGS)


@polars_available
def test_objects_are_allclose_polars_opposite_directions() -> None:
    assert objects_are_allclose(pl.Series(OPPOSITE_X), pl.Series(OPPOSITE_Y), **KWARGS)

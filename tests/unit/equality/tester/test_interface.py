from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence

import pytest

from coola.equality.config import EqualityConfig2
from coola.equality.tester import (
    DefaultEqualityTester,
    EqualityTesterRegistry,
    JaxArrayEqualityTester,
    MappingEqualityTester,
    NumpyArrayEqualityTester,
    PandasDataFrameEqualityTester,
    PandasSeriesEqualityTester,
    PolarsDataFrameEqualityTester,
    PolarsLazyFrameEqualityTester,
    PolarsSeriesEqualityTester,
    PyarrowEqualityTester,
    ScalarEqualityTester,
    SequenceEqualityTester,
    TorchPackedSequenceEqualityTester,
    TorchTensorEqualityTester,
    XarrayDataArrayEqualityTester,
    XarrayDatasetEqualityTester,
    XarrayVariableEqualityTester,
    get_default_registry,
    register_equality_testers,
)
from coola.equality.tester.jax import get_array_impl_class
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


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


@pytest.fixture
def config() -> EqualityConfig2:
    return EqualityConfig2()


class CustomList(list):
    r"""Create a custom class that inherits from list."""


###############################################
#     Tests for register_equality_testers     #
###############################################


def test_register_equality_testers_calls_registry() -> None:
    register_equality_testers({CustomList: SequenceEqualityTester()})
    assert get_default_registry().has_equality_tester(CustomList)


def test_register_equality_testers_with_exist_ok_true() -> None:
    register_equality_testers({CustomList: DefaultEqualityTester()})
    register_equality_testers({CustomList: SequenceEqualityTester()}, exist_ok=True)


def test_register_equality_testers_with_exist_ok_false() -> None:
    register_equality_testers({CustomList: DefaultEqualityTester()})
    with pytest.raises(RuntimeError, match="already registered"):
        register_equality_testers({CustomList: SequenceEqualityTester()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a EqualityTesterRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, EqualityTesterRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same instance on
    multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


@pytest.mark.parametrize("dtype", [object, str, complex, set, frozenset])
def test_get_default_registry_default(dtype: type) -> None:
    """Test that scalar types are registered with
    DefaultEqualityTester."""
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), DefaultEqualityTester)


@pytest.mark.parametrize("dtype", [int, float, bool])
def test_get_default_registry_scalars(dtype: type) -> None:
    """Test that scalar types are registered with
    DefaultEqualityTester."""
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), ScalarEqualityTester)


@pytest.mark.parametrize("dtype", [list, tuple, Sequence])
def test_get_default_registry_sequences(dtype: type) -> None:
    """Test that sequence types are registered with
    SequenceEqualityTester."""
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), SequenceEqualityTester)


@pytest.mark.parametrize("dtype", [dict, Mapping])
def test_get_default_registry_mappings(dtype: type) -> None:
    """Test that mapping types are registered with
    MappingEqualityTester."""
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), MappingEqualityTester)


@jax_available
def test_get_default_registry_jax_ndarray() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(jnp.ndarray)
    assert isinstance(registry.find_equality_tester(jnp.ndarray), JaxArrayEqualityTester)


@jax_available
def test_get_default_registry_jax_array() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(get_array_impl_class())
    assert isinstance(registry.find_equality_tester(get_array_impl_class()), JaxArrayEqualityTester)


@numpy_available
def test_get_default_registry_numpy_ndarray() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(np.ndarray)
    assert isinstance(registry.find_equality_tester(np.ndarray), NumpyArrayEqualityTester)


@pandas_available
def test_get_default_registry_pandas_dataframe() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(pd.DataFrame)
    assert isinstance(registry.find_equality_tester(pd.DataFrame), PandasDataFrameEqualityTester)


@pandas_available
def test_get_default_registry_pandas_series() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(pd.Series)
    assert isinstance(registry.find_equality_tester(pd.Series), PandasSeriesEqualityTester)


@polars_available
def test_get_default_registry_polars_dataframe() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(pl.DataFrame)
    assert isinstance(registry.find_equality_tester(pl.DataFrame), PolarsDataFrameEqualityTester)


@polars_available
def test_get_default_registry_polars_lazyframe() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(pl.LazyFrame)
    assert isinstance(registry.find_equality_tester(pl.LazyFrame), PolarsLazyFrameEqualityTester)


@polars_available
def test_get_default_registry_polars_series() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(pl.Series)
    assert isinstance(registry.find_equality_tester(pl.Series), PolarsSeriesEqualityTester)


@pyarrow_available
def test_get_default_registry_pyarrow_array() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(pa.Array)
    assert isinstance(registry.find_equality_tester(pa.Array), PyarrowEqualityTester)


@pyarrow_available
def test_get_default_registry_pyarrow_table() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(pa.Table)
    assert isinstance(registry.find_equality_tester(pa.Table), PyarrowEqualityTester)


@torch_available
def test_get_default_registry_torch_packed_sequence() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(torch.nn.utils.rnn.PackedSequence)
    assert isinstance(
        registry.find_equality_tester(torch.nn.utils.rnn.PackedSequence),
        TorchPackedSequenceEqualityTester,
    )


@torch_available
def test_get_default_registry_torch_tensor() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(torch.Tensor)
    assert isinstance(registry.find_equality_tester(torch.Tensor), TorchTensorEqualityTester)


@xarray_available
def test_get_default_registry_xarray_data_array() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(xr.DataArray)
    assert isinstance(registry.find_equality_tester(xr.DataArray), XarrayDataArrayEqualityTester)


@xarray_available
def test_get_default_registry_xarray_dataset() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(xr.Dataset)
    assert isinstance(registry.find_equality_tester(xr.Dataset), XarrayDatasetEqualityTester)


@xarray_available
def test_get_default_registry_xarray_variable() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(xr.Variable)
    assert isinstance(registry.find_equality_tester(xr.Variable), XarrayVariableEqualityTester)


def test_get_default_registry_objects_are_equal_true(config: EqualityConfig2) -> None:
    """Test that default registry can transform a list."""
    registry = get_default_registry()
    assert registry.objects_are_equal([1, 2, 3], [1, 2, 3], config=config)


def test_get_default_registry_objects_are_equal_false(config: EqualityConfig2) -> None:
    """Test that default registry can transform a list."""
    registry = get_default_registry()
    assert not registry.objects_are_equal([1, 2, 3], [1, 1], config=config)


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist across calls."""
    registry1 = get_default_registry()
    assert not registry1.has_equality_tester(CustomList)
    registry1.register(CustomList, SequenceEqualityTester())
    assert registry1.has_equality_tester(CustomList)

    # Get registry again
    registry2 = get_default_registry()
    assert registry1 is registry2
    assert registry2.has_equality_tester(CustomList)

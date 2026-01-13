from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Generator, Mapping, Sequence
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
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
else:
    jnp = Mock()

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_pandas_available():
    import pandas as pd
else:
    pd = Mock()

if is_polars_available():
    import polars as pl
else:
    pl = Mock()

if is_pyarrow_available():
    import pyarrow as pa
else:
    pa = Mock()

if is_torch_available():
    import torch
else:
    torch = Mock()

if is_xarray_available():
    import xarray as xr
else:
    xr = Mock()


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


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


@pytest.mark.parametrize(
    ("dtype", "tester_class", "is_registered"),
    [
        # Default
        (object, DefaultEqualityTester, True),
        (str, DefaultEqualityTester, False),
        (complex, DefaultEqualityTester, False),
        (set, DefaultEqualityTester, False),
        (frozenset, DefaultEqualityTester, False),
        # Scalar
        (int, ScalarEqualityTester, True),
        (float, ScalarEqualityTester, True),
        (bool, ScalarEqualityTester, False),
        # Sequence
        (list, SequenceEqualityTester, True),
        (tuple, SequenceEqualityTester, True),
        (Sequence, SequenceEqualityTester, True),
        (deque, SequenceEqualityTester, True),
        # Mapping
        (dict, MappingEqualityTester, True),
        (Mapping, MappingEqualityTester, True),
        (OrderedDict, MappingEqualityTester, False),
    ],
)
def test_get_default_registry_native(dtype: type, tester_class: type, is_registered: bool) -> None:
    registry = get_default_registry()
    if is_registered:
        assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), tester_class)


@jax_available
@pytest.mark.parametrize(
    ("dtype", "tester_class"),
    [
        (jnp.ndarray, JaxArrayEqualityTester),
        (get_array_impl_class(), JaxArrayEqualityTester),
    ],
)
def test_get_default_registry_jax(dtype: type, tester_class: type) -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), tester_class)


@numpy_available
def test_get_default_registry_numpy_ndarray() -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(np.ndarray)
    assert isinstance(registry.find_equality_tester(np.ndarray), NumpyArrayEqualityTester)


@pandas_available
@pytest.mark.parametrize(
    ("dtype", "tester_class"),
    [
        (pd.DataFrame, PandasDataFrameEqualityTester),
        (pd.Series, PandasSeriesEqualityTester),
    ],
)
def test_get_default_registry_pandas(dtype: type, tester_class: type) -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), tester_class)


@polars_available
@pytest.mark.parametrize(
    ("dtype", "tester_class"),
    [
        (pl.DataFrame, PolarsDataFrameEqualityTester),
        (pl.LazyFrame, PolarsLazyFrameEqualityTester),
        (pl.Series, PolarsSeriesEqualityTester),
    ],
)
def test_get_default_registry_polars(dtype: type, tester_class: type) -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), tester_class)


@pyarrow_available
@pytest.mark.parametrize(
    ("dtype", "tester_class"),
    [
        (pa.Array, PyarrowEqualityTester),
        (pa.Table, PyarrowEqualityTester),
    ],
)
def test_get_default_registry_pyarrow(dtype: type, tester_class: type) -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), tester_class)


@torch_available
@pytest.mark.parametrize(
    ("dtype", "tester_class"),
    [
        (torch.nn.utils.rnn.PackedSequence, TorchPackedSequenceEqualityTester),
        (torch.Tensor, TorchTensorEqualityTester),
    ],
)
def test_get_default_registry_torch(dtype: type, tester_class: type) -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), tester_class)


@xarray_available
@pytest.mark.parametrize(
    ("dtype", "tester_class"),
    [
        (xr.DataArray, XarrayDataArrayEqualityTester),
        (xr.Dataset, XarrayDatasetEqualityTester),
        (xr.Variable, XarrayVariableEqualityTester),
    ],
)
def test_get_default_registry_xarray(dtype: type, tester_class: type) -> None:
    registry = get_default_registry()
    assert registry.has_equality_tester(dtype)
    assert isinstance(registry.find_equality_tester(dtype), tester_class)


def test_get_default_registry_objects_are_equal_true(config: EqualityConfig) -> None:
    """Test that default registry can transform a list."""
    registry = get_default_registry()
    assert registry.objects_are_equal([1, 2, 3], [1, 2, 3], config=config)


def test_get_default_registry_objects_are_equal_false(config: EqualityConfig) -> None:
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

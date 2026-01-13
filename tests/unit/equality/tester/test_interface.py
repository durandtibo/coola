from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Generator, Mapping, Sequence

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
def config() -> EqualityConfig:
    return EqualityConfig()


class CustomList(list):
    r"""Create a custom class that inherits from list."""


TESTER_TYPES: list[tuple[type, type, bool]] = [
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
]
if is_jax_available():
    TESTER_TYPES.extend(
        [
            (jnp.ndarray, JaxArrayEqualityTester, True),
            (get_array_impl_class(), JaxArrayEqualityTester, True),
        ]
    )
if is_numpy_available():
    TESTER_TYPES.append((np.ndarray, NumpyArrayEqualityTester, True))
if is_pandas_available():
    TESTER_TYPES.extend(
        [
            (pd.DataFrame, PandasDataFrameEqualityTester, True),
            (pd.Series, PandasSeriesEqualityTester, True),
        ]
    )
if is_polars_available():
    TESTER_TYPES.extend(
        [
            (pl.DataFrame, PolarsDataFrameEqualityTester, True),
            (pl.LazyFrame, PolarsLazyFrameEqualityTester, True),
            (pl.Series, PolarsSeriesEqualityTester, True),
        ]
    )
if is_pyarrow_available():
    TESTER_TYPES.extend(
        [
            (pa.Array, PyarrowEqualityTester, True),
            (pa.Table, PyarrowEqualityTester, True),
        ],
    )
if is_torch_available():
    TESTER_TYPES.extend(
        [
            (torch.nn.utils.rnn.PackedSequence, TorchPackedSequenceEqualityTester, True),
            (torch.Tensor, TorchTensorEqualityTester, True),
        ],
    )
if is_xarray_available():
    TESTER_TYPES.extend(
        [
            (xr.DataArray, XarrayDataArrayEqualityTester, True),
            (xr.Dataset, XarrayDatasetEqualityTester, True),
            (xr.Variable, XarrayVariableEqualityTester, True),
        ]
    )

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


@pytest.mark.parametrize(("dtype", "tester_class", "is_registered"), TESTER_TYPES)
def test_get_default_registry_types(dtype: type, tester_class: type, is_registered: bool) -> None:
    registry = get_default_registry()
    if is_registered:
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

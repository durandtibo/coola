from __future__ import annotations

import pkgutil
from array import array
from collections import OrderedDict, deque
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester import (
    BaseEqualityTester,
    DataclassEqualityTester,
    DefaultEqualityTester,
    EqualityTesterRegistry,
    JaxArrayEqualityTester,
    MappingEqualityTester,
    NumpyArrayEqualityTester,
    OrderedMappingEqualityTester,
    ScalarEqualityTester,
    SequenceEqualityTester,
    SetEqualityTester,
    TableEqualityTester,
    TorchPackedSequenceEqualityTester,
    TorchTensorEqualityTester,
    XarrayDataArrayEqualityTester,
    XarrayDatasetEqualityTester,
    XarrayVariableEqualityTester,
    get_default_registry,
    interface as tester_interface,
    register_equality_testers,
)
from coola.io import TextLoader, TextSaver
from coola.io.base import BaseLoader, BaseSaver
from coola.registry import AlreadyRegisteredError
from coola.testing.marks import numpy_available
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
    tester_interface._default_registry.reset()
    yield
    tester_interface._default_registry.reset()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


class CustomList(list):
    r"""Create a custom class that inherits from list."""


TESTER_TYPES: list[tuple[type, type, bool]] = [
    # Default
    (object, DefaultEqualityTester, True),
    (str, DefaultEqualityTester, False),
    (set, SetEqualityTester, True),
    (frozenset, SetEqualityTester, True),
    # Scalar
    (int, ScalarEqualityTester, True),
    (float, ScalarEqualityTester, True),
    (complex, ScalarEqualityTester, True),
    (bool, ScalarEqualityTester, False),
    # Sequence
    (list, SequenceEqualityTester, True),
    (tuple, SequenceEqualityTester, True),
    (Sequence, SequenceEqualityTester, True),
    (deque, SequenceEqualityTester, True),
    (array, SequenceEqualityTester, True),
    (type({}.values()), SequenceEqualityTester, True),
    # Mapping
    (dict, MappingEqualityTester, True),
    (Mapping, MappingEqualityTester, True),
    (OrderedDict, OrderedMappingEqualityTester, True),
    (MappingProxyType, MappingEqualityTester, True),
    # coola types: unregistered, compared with ``==``
    (BaseEqualityTester, DefaultEqualityTester, False),
]
if is_jax_available():
    TESTER_TYPES.extend(
        [
            (jnp.ndarray, JaxArrayEqualityTester, True),
            (type(jnp.ones(1)), JaxArrayEqualityTester, False),
        ]
    )
if is_numpy_available():
    TESTER_TYPES.append((np.ndarray, NumpyArrayEqualityTester, True))
if is_pandas_available():
    TESTER_TYPES.extend(
        [
            (pd.DataFrame, TableEqualityTester, True),
            (pd.Series, TableEqualityTester, True),
            (pd.Index, TableEqualityTester, True),
            (pd.MultiIndex, TableEqualityTester, False),
            (pd.Categorical, TableEqualityTester, False),
        ]
    )
if is_polars_available():
    TESTER_TYPES.extend(
        [
            (pl.DataFrame, TableEqualityTester, True),
            (pl.LazyFrame, TableEqualityTester, True),
            (pl.Series, TableEqualityTester, True),
        ]
    )
if is_pyarrow_available():
    TESTER_TYPES.extend(
        [
            (pa.Array, TableEqualityTester, True),
            (pa.ChunkedArray, TableEqualityTester, True),
            (pa.RecordBatch, TableEqualityTester, True),
            (pa.Table, TableEqualityTester, True),
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


def test_register_equality_testers_exist_ok_is_keyword_only() -> None:
    positional = (True,)
    with pytest.raises(TypeError, match="positional"):
        register_equality_testers({CustomList: SequenceEqualityTester()}, *positional)  # ty: ignore[too-many-positional-arguments]


def test_register_equality_testers_with_exist_ok_false() -> None:
    register_equality_testers({CustomList: DefaultEqualityTester()})
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
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
    "dtype", [dtype for dtype, _, is_registered in TESTER_TYPES if is_registered]
)
def test_get_default_registry_has_the_registered_types(dtype: type) -> None:
    assert get_default_registry().has_equality_tester(dtype)


@pytest.mark.parametrize(("dtype", "tester_class"), [(t[0], t[1]) for t in TESTER_TYPES])
def test_get_default_registry_types(dtype: type, tester_class: type) -> None:
    assert isinstance(get_default_registry().find_equality_tester(dtype), tester_class)


def test_get_default_registry_singleton_does_not_have_the_custom_type() -> None:
    """Test that the registry does not have the custom type."""
    assert not get_default_registry().has_equality_tester(CustomList)


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist across calls."""
    registry1 = get_default_registry()
    registry1.register(CustomList, SequenceEqualityTester())
    assert registry1.has_equality_tester(CustomList)


def test_get_default_registry_singleton_returns_the_same_registry() -> None:
    """Test that the registry is the same across calls."""
    registry1 = get_default_registry()
    registry1.register(CustomList, SequenceEqualityTester())
    assert registry1 is get_default_registry()


def test_get_default_registry_singleton_persists_modifications_across_calls() -> None:
    """Test that modifications to the registry persist across calls."""
    get_default_registry().register(CustomList, SequenceEqualityTester())
    assert get_default_registry().has_equality_tester(CustomList)


@numpy_available
@patch("coola.equality.tester.interface.package_available", lambda name: False)  # noqa: ARG005
def test_register_default_equality_testers_skips_the_libraries_that_are_not_installed() -> None:
    registry = EqualityTesterRegistry()
    tester_interface._register_default_equality_testers(registry)
    assert not registry.has(np.ndarray)


#############################################
#     Tests for the lazy equality testers     #
#############################################

LAZY_MODULES = [
    pytest.param("coola.equality.tester.jax", ("jax", "jaxlib"), id="jax"),
    pytest.param("coola.equality.tester.numpy", ("numpy",), id="numpy"),
    pytest.param("coola.equality.tester.pandas", ("pandas",), id="pandas"),
    pytest.param("coola.equality.tester.polars", ("polars",), id="polars"),
    pytest.param("coola.equality.tester.pyarrow", ("pyarrow",), id="pyarrow"),
    pytest.param("coola.equality.tester.torch", ("torch",), id="torch"),
    pytest.param("coola.equality.tester.xarray", ("xarray",), id="xarray"),
]

LAZY_MODULE_NAMES = [
    "coola.equality.tester.jax",
    "coola.equality.tester.numpy",
    "coola.equality.tester.pandas",
    "coola.equality.tester.polars",
    "coola.equality.tester.pyarrow",
    "coola.equality.tester.torch",
    "coola.equality.tester.xarray",
]


@pytest.mark.parametrize(("module", "names"), LAZY_MODULES)
def test_get_default_testers_types_come_from_the_library(
    module: str, names: tuple[str, ...]
) -> None:
    """Test that the lazy group is loaded for each type that it
    registers."""
    pytest.importorskip(names[0])
    mapping = pkgutil.resolve_name(f"{module}:get_default_testers")()
    assert {t.__module__.partition(".")[0] for t in mapping} <= set(names)


@pytest.mark.parametrize(("module", "names"), LAZY_MODULES)
def test_get_default_testers_returns_entries_if_the_library_is_available(
    module: str, names: tuple[str, ...]
) -> None:
    pytest.importorskip(names[0])
    assert len(pkgutil.resolve_name(f"{module}:get_default_testers")()) > 0


@pytest.mark.parametrize("module", LAZY_MODULE_NAMES)
def test_get_default_testers_returns_nothing_if_the_library_is_unavailable(module: str) -> None:
    name = module.rpartition(".")[2]
    with patch(f"{module}.is_{name}_available", lambda: False):
        assert pkgutil.resolve_name(f"{module}:get_default_testers")() == {}


######################################################
#     Tests for the testers of the default types     #
######################################################


@dataclass
class Point:
    x: int = 1


# One sample for each type with a tester that does not need an optional library.
# A new native type must be added here, so it is checked to be reflexive.
NATIVE_SAMPLES: dict[type, Any] = {
    object: object(),
    int: 1,
    float: 1.5,
    complex: 1 + 2j,
    Decimal: Decimal("1.5"),
    Fraction: Fraction(1, 3),
    timedelta: timedelta(seconds=1),
    deque: deque([1, 2]),
    array: array("d", [1.0, 2.0]),
    list: [1, {"a": 2}],
    tuple: (1, (2, 3)),
    Sequence: [1, 2],
    set: {1, 2},
    frozenset: frozenset({1, 2}),
    type({}.values()): {"a": 1}.values(),
    dict: {"a": [1, 2]},
    OrderedDict: OrderedDict(a=1, b=2),
    MappingProxyType: MappingProxyType({"a": [1, 2]}),
    Mapping: {"a": 1},
    BaseLoader: TextLoader(),
    BaseSaver: TextSaver(),
}


@pytest.mark.parametrize("data_type", NATIVE_SAMPLES)
def test_default_tester_is_reflexive(data_type: type) -> None:
    sample = NATIVE_SAMPLES[data_type]
    assert Engine(EqualityConfig()).run(sample, sample)


@pytest.mark.parametrize("data_type", NATIVE_SAMPLES)
def test_default_tester_equals_itself(data_type: type) -> None:
    tester = get_default_registry().resolve(data_type)
    assert tester == tester


def test_default_registry_native_types_have_a_sample() -> None:
    optional = ("jax", "numpy", "pandas", "polars", "pyarrow", "pydantic", "torch", "xarray")
    native = {
        data_type
        for data_type, _ in get_default_registry().items()
        if not data_type.__module__.startswith(optional)
    }
    assert native == set(NATIVE_SAMPLES)


def test_default_registry_dataclass_is_reflexive() -> None:
    assert Engine(EqualityConfig()).run(Point(), Point())


def test_default_registry_dataclass_uses_the_dataclass_tester() -> None:
    assert isinstance(get_default_registry().resolve(Point), DataclassEqualityTester)


def test_default_registry_has_the_dataclass_fallback() -> None:
    assert [handler for _, handler in get_default_registry().fallbacks()] == [
        DataclassEqualityTester()
    ]

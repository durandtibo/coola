from __future__ import annotations

import pkgutil
from collections import UserString, deque
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock, patch

import pytest

from coola.registry import AlreadyRegisteredError
from coola.summary import (
    DataclassSummarizer,
    DefaultSummarizer,
    JaxArraySummarizer,
    MappingSummarizer,
    NDArraySummarizer,
    PandasDataFrameSummarizer,
    PandasSeriesSummarizer,
    PolarsDataFrameSummarizer,
    PolarsLazyFrameSummarizer,
    PolarsSeriesSummarizer,
    PyarrowArraySummarizer,
    PyarrowTableSummarizer,
    SequenceSummarizer,
    SetSummarizer,
    SummarizerRegistry,
    TensorSummarizer,
    XarrayDataArraySummarizer,
    XarrayDatasetSummarizer,
    XarrayVariableSummarizer,
    get_default_registry,
    interface as summary_interface,
    register_summarizers,
    summarize,
)
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

if is_torch_available():  # pragma: no cover
    import torch

if is_numpy_available():  # pragma: no cover
    import numpy as np

if is_jax_available():  # pragma: no cover
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()

if is_pandas_available():  # pragma: no cover
    import pandas as pd
else:  # pragma: no cover
    pd = Mock()

if is_polars_available():  # pragma: no cover
    import polars as pl
else:  # pragma: no cover
    pl = Mock()

if is_pyarrow_available():  # pragma: no cover
    import pyarrow as pa
else:  # pragma: no cover
    pa = Mock()

if is_xarray_available():  # pragma: no cover
    import xarray as xr
else:  # pragma: no cover
    xr = Mock()


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    summary_interface._default_registry.reset()
    yield
    summary_interface._default_registry.reset()


class CustomList(list):
    r"""Create a custom class that inherits from list."""


###############################
#     Tests for summarize     #
###############################


def test_summarize_with_simple_value() -> None:
    assert summarize(5) == "<class 'int'> 5"


def test_summarize_with_string() -> None:
    assert summarize("hello") == "<class 'str'> hello"


def test_summarize_with_dict() -> None:
    assert summarize({"a": 1, "b": 2}) == "<class 'dict'> (length=2)\n  (a): 1\n  (b): 2"


def test_summarize_with_list() -> None:
    assert summarize([1, 2, 3]) == "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_summarize_with_max_depth_1() -> None:
    assert (
        summarize({"level1": {"level2": {"level3": [1, 2, 3]}}})
        == "<class 'dict'> (length=1)\n  (level1): {'level2': {'level3': [1, 2, 3]}}"
    )


def test_summarize_with_max_depth_2() -> None:
    assert (
        summarize({"level1": {"level2": {"level3": [1, 2, 3]}}}, max_depth=2)
        == "<class 'dict'> (length=1)\n"
        "  (level1): <class 'dict'> (length=1)\n"
        "    (level2): {'level3': [1, 2, 3]}"
    )


def test_summarize_with_max_depth_3() -> None:
    assert (
        summarize({"level1": {"level2": {"level3": [1, 2, 3]}}}, max_depth=3)
        == "<class 'dict'> (length=1)\n"
        "  (level1): <class 'dict'> (length=1)\n"
        "    (level2): <class 'dict'> (length=1)\n"
        "      (level3): [1, 2, 3]"
    )


def test_summarize_with_custom_registry() -> None:
    assert (
        summarize({"a": 1, "b": 2}, registry=SummarizerRegistry({object: DefaultSummarizer()}))
        == "<class 'dict'> {'a': 1, 'b': 2}"
    )


##########################################
#     Tests for register_summarizers     #
##########################################


def test_register_summarizers_calls_registry() -> None:
    register_summarizers({CustomList: SequenceSummarizer()})
    assert get_default_registry().has_summarizer(CustomList)


def test_register_summarizers_with_exist_ok_true() -> None:
    register_summarizers({CustomList: SetSummarizer()})
    register_summarizers({CustomList: SequenceSummarizer()}, exist_ok=True)


def test_register_summarizers_with_exist_ok_false() -> None:
    register_summarizers({CustomList: SetSummarizer()})
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        register_summarizers({CustomList: SequenceSummarizer()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a SummarizerRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, SummarizerRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same instance on
    multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


@dataclass
class _Point:
    x: int = 1


def test_get_default_registry_dataclass_summarizer() -> None:
    assert isinstance(get_default_registry().find_summarizer(_Point), DataclassSummarizer)


def test_get_default_registry_has_the_dataclass_fallback() -> None:
    assert [type(handler) for _, handler in get_default_registry().fallbacks()] == [
        DataclassSummarizer
    ]


def test_get_default_registry_object_is_registered() -> None:
    assert get_default_registry().has_summarizer(object)


@pytest.mark.parametrize("dtype", [object, int, float, str, bool, complex])
def test_get_default_registry_default_summarizer(dtype: type) -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(dtype), DefaultSummarizer)


@pytest.mark.parametrize("dtype", [list, tuple, deque, Sequence])
def test_get_default_registry_sequences_is_registered(dtype: type) -> None:
    """Test that sequence types are registered with
    SequenceSummarizer."""
    registry = get_default_registry()
    assert registry.has_summarizer(dtype)


@pytest.mark.parametrize("dtype", [list, tuple, deque, Sequence])
def test_get_default_registry_sequences_summarizer(dtype: type) -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(dtype), SequenceSummarizer)


@pytest.mark.parametrize("dtype", [set, frozenset])
def test_get_default_registry_sets_is_registered(dtype: type) -> None:
    """Test that set types are registered with SetSummarizer."""
    registry = get_default_registry()
    assert registry.has_summarizer(dtype)


@pytest.mark.parametrize("dtype", [set, frozenset])
def test_get_default_registry_sets_summarizer(dtype: type) -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(dtype), SetSummarizer)


@pytest.mark.parametrize("dtype", [dict, Mapping])
def test_register_default_summarizers_registers_mappings_is_registered(dtype: type) -> None:
    """Test that mapping types are registered with MappingSummarizer."""
    registry = get_default_registry()
    assert registry.has_summarizer(dtype)


@pytest.mark.parametrize("dtype", [dict, Mapping])
def test_register_default_summarizers_registers_mappings_summarizer(dtype: type) -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(dtype), MappingSummarizer)


@numpy_available
def test_register_default_summarizers_registers_ndarray_is_registered() -> None:
    registry = get_default_registry()
    assert registry.has_summarizer(np.ndarray)


@numpy_available
def test_register_default_summarizers_registers_ndarray_summarizer() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(np.ndarray), NDArraySummarizer)


@torch_available
def test_register_default_summarizers_registers_tensor_is_registered() -> None:
    registry = get_default_registry()
    assert registry.has_summarizer(torch.Tensor)


@torch_available
def test_register_default_summarizers_registers_tensor_summarizer() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(torch.Tensor), TensorSummarizer)


def test_default_registry_can_summarize_list() -> None:
    """Test that default registry can summarize a list."""
    registry = get_default_registry()
    assert (
        registry.summarize([1, 2, 3]) == "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"
    )


def test_default_registry_can_summarize_dict() -> None:
    """Test that default registry can summarize a dict."""
    registry = get_default_registry()
    assert registry.summarize({"a": 1, "b": 2}) == "<class 'dict'> (length=2)\n  (a): 1\n  (b): 2"


def test_get_default_registry_singleton_does_not_have_the_custom_type() -> None:
    """Test that the registry does not have the custom type."""
    assert not get_default_registry().has_summarizer(CustomList)


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist."""
    registry1 = get_default_registry()
    registry1.register(CustomList, SequenceSummarizer())
    assert registry1.has_summarizer(CustomList)


def test_get_default_registry_singleton_returns_the_same_registry() -> None:
    """Test that the registry is the same across calls."""
    registry1 = get_default_registry()
    registry1.register(CustomList, SequenceSummarizer())
    assert registry1 is get_default_registry()


def test_get_default_registry_singleton_persists_modifications_across_calls() -> None:
    """Test that modifications to the registry persist across calls."""
    get_default_registry().register(CustomList, SequenceSummarizer())
    assert get_default_registry().has_summarizer(CustomList)


@pandas_available
def test_get_default_registry_pandas_summarizers_dataframe() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(pd.DataFrame), PandasDataFrameSummarizer)


@pandas_available
def test_get_default_registry_pandas_summarizers_series() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(pd.Series), PandasSeriesSummarizer)


@polars_available
def test_get_default_registry_polars_summarizers_dataframe() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(pl.DataFrame), PolarsDataFrameSummarizer)


@polars_available
def test_get_default_registry_polars_summarizers_lazyframe() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(pl.LazyFrame), PolarsLazyFrameSummarizer)


@polars_available
def test_get_default_registry_polars_summarizers_series() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(pl.Series), PolarsSeriesSummarizer)


@pyarrow_available
def test_get_default_registry_pyarrow_summarizers_array() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(pa.Array), PyarrowArraySummarizer)


@pyarrow_available
def test_get_default_registry_pyarrow_summarizers_chunked_array() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(pa.ChunkedArray), PyarrowArraySummarizer)


@pyarrow_available
def test_get_default_registry_pyarrow_summarizers_table() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(pa.Table), PyarrowTableSummarizer)


@xarray_available
def test_get_default_registry_xarray_summarizers_data_array() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(xr.DataArray), XarrayDataArraySummarizer)


@xarray_available
def test_get_default_registry_xarray_summarizers_dataset() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(xr.Dataset), XarrayDatasetSummarizer)


@xarray_available
def test_get_default_registry_xarray_summarizers_variable() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(xr.Variable), XarrayVariableSummarizer)


@jax_available
def test_get_default_registry_jax_summarizers_ndarray() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(jnp.ndarray), JaxArraySummarizer)


@jax_available
def test_get_default_registry_jax_summarizers_array_type() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_summarizer(type(jnp.ones(1))), JaxArraySummarizer)


@pandas_available
@polars_available
@pyarrow_available
@xarray_available
@jax_available
def test_summarize_nested_structure_with_dataframes_pandas() -> None:
    data = {"pandas": pd.DataFrame({"a": [1]}), "polars": pl.DataFrame({"a": [1]})}
    result = summarize(data, max_depth=2)
    assert "pandas.DataFrame'> | shape=(1, 1) | columns=a: int64" in result


@pandas_available
@polars_available
@pyarrow_available
@xarray_available
@jax_available
def test_summarize_nested_structure_with_dataframes_polars() -> None:
    data = {"pandas": pd.DataFrame({"a": [1]}), "polars": pl.DataFrame({"a": [1]})}
    result = summarize(data, max_depth=2)
    assert "shape=(1, 1) | columns=a: Int64" in result


@pytest.mark.parametrize(
    "data_type",
    [
        pytest.param(jnp.ndarray, id="jax"),
        pytest.param(pd.DataFrame, id="pandas"),
        pytest.param(pl.DataFrame, id="polars"),
        pytest.param(pa.Table, id="pyarrow"),
        pytest.param(xr.Dataset, id="xarray"),
    ],
)
@jax_available
@pandas_available
@polars_available
@pyarrow_available
@xarray_available
@patch("coola.summary.interface.package_available", lambda _: False)
def test_register_default_summarizers_without_optional_package(data_type: type) -> None:
    registry = SummarizerRegistry()
    summary_interface._register_default_summarizers(registry)
    assert not registry.has(data_type)


def test_summarize_negative_max_depth_raises() -> None:
    with pytest.raises(ValueError, match="max_depth must be greater than or equal to 0"):
        summarize([1, 2], max_depth=-1)


##########################################
#     Tests for the lazy summarizers     #
##########################################

LAZY_MODULES = [
    pytest.param("coola.summary.jax", ("jax", "jaxlib"), id="jax"),
    pytest.param("coola.summary.numpy", ("numpy",), id="numpy"),
    pytest.param("coola.summary.pandas", ("pandas",), id="pandas"),
    pytest.param("coola.summary.polars", ("polars",), id="polars"),
    pytest.param("coola.summary.pyarrow", ("pyarrow",), id="pyarrow"),
    pytest.param("coola.summary.torch", ("torch",), id="torch"),
    pytest.param("coola.summary.xarray", ("xarray",), id="xarray"),
]

LAZY_MODULE_NAMES = [
    "coola.summary.jax",
    "coola.summary.numpy",
    "coola.summary.pandas",
    "coola.summary.polars",
    "coola.summary.pyarrow",
    "coola.summary.torch",
    "coola.summary.xarray",
]


@pytest.mark.parametrize(("module", "names"), LAZY_MODULES)
def test_get_default_summarizers_types_come_from_the_library(
    module: str, names: tuple[str, ...]
) -> None:
    """Test that the lazy group is loaded for each type that it
    registers."""
    pytest.importorskip(names[0])
    mapping = pkgutil.resolve_name(f"{module}:get_default_summarizers")()
    assert {t.__module__.partition(".")[0] for t in mapping} <= set(names)


@pytest.mark.parametrize(("module", "names"), LAZY_MODULES)
def test_get_default_summarizers_returns_entries_if_the_library_is_available(
    module: str, names: tuple[str, ...]
) -> None:
    pytest.importorskip(names[0])
    assert len(pkgutil.resolve_name(f"{module}:get_default_summarizers")()) > 0


@pytest.mark.parametrize("module", LAZY_MODULE_NAMES)
def test_get_default_summarizers_returns_nothing_if_the_library_is_unavailable(module: str) -> None:
    name = module.rpartition(".")[2]
    with patch(f"{module}.is_{name}_available", lambda: False):
        assert pkgutil.resolve_name(f"{module}:get_default_summarizers")() == {}


def test_summarize_positional_max_depth_raises() -> None:
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        summarize([1], 2)  # ty: ignore[too-many-positional-arguments]


def test_summarize_positional_registry_raises() -> None:
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        summarize([1], 1, None)  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize("max_depth", [True, False, 1.5, "1", None])
def test_summarize_max_depth_not_int_raises(max_depth: object) -> None:
    with pytest.raises(TypeError, match="max_depth"):
        summarize([1], max_depth=max_depth)  # ty: ignore[invalid-argument-type]


def test_register_summarizers_positional_exist_ok_raises() -> None:
    with pytest.raises(TypeError, match="takes 1 positional argument"):
        register_summarizers({}, True)  # ty: ignore[too-many-positional-arguments]


###############################################
#     Tests for the dispatch of the types     #
###############################################


def test_get_default_registry_user_string_summarizer() -> None:
    assert isinstance(get_default_registry().find_summarizer(UserString), DefaultSummarizer)


def test_get_default_registry_mapping_proxy_summarizer() -> None:
    assert isinstance(get_default_registry().find_summarizer(MappingProxyType), MappingSummarizer)


def test_summarize_user_string() -> None:
    assert summarize(UserString("abc")) == f"{UserString} abc"


def test_summarize_mapping_proxy() -> None:
    assert summarize(MappingProxyType({"a": 1, "b": 2})) == (
        f"{MappingProxyType} (length=2)\n  (a): 1\n  (b): 2"
    )


def test_summarize_user_string_in_list() -> None:
    assert summarize([UserString("abc")]) == f"{list} (length=1)\n  (0): abc"


##############################################
#     Tests for the options of summarize     #
##############################################


def test_summarize_max_items() -> None:
    assert summarize(list(range(10)), max_items=2) == (
        "<class 'list'> (length=10)\n  (0): 0\n  (1): 1\n  ... (+8 more)"
    )


def test_summarize_max_items_none_shows_all_items() -> None:
    assert summarize(list(range(7)), max_items=None).count("\n") == 7


def test_summarize_max_items_applies_to_every_container_type() -> None:
    assert summarize({"a": 1, "b": 2, "c": 3}, max_items=1) == (
        "<class 'dict'> (length=3)\n  (a): 1\n  ... (+2 more)"
    )


def test_summarize_max_characters() -> None:
    assert summarize("abcdef", max_characters=4) == "<cla..."


def test_summarize_max_characters_none_does_not_cut() -> None:
    assert summarize("a" * 1500, max_characters=None) == "<class 'str'> " + "a" * 1500


def test_summarize_show_data_with_a_container() -> None:
    assert summarize([1], show_data=True) == "<class 'list'> (length=1)\n  (0): 1"


def test_summarize_with_a_custom_registry() -> None:
    registry = SummarizerRegistry({object: DefaultSummarizer()})
    assert summarize([1], registry=registry) == "<class 'list'> [1]"


@pytest.mark.parametrize("name", ["max_items", "max_characters"])
def test_summarize_negative_limit_raises(name: str) -> None:
    kwargs: dict[str, Any] = {name: -1}
    with pytest.raises(ValueError, match=f"{name} must be greater than or equal to 0"):
        summarize([1], **kwargs)


def test_summarize_show_data_not_bool_raises() -> None:
    with pytest.raises(TypeError, match="show_data"):
        summarize([1], show_data="yes")  # ty: ignore[invalid-argument-type]

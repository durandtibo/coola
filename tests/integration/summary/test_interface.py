from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import UserString, deque
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

import coola
from coola.summary import Node, SummaryConfig, get_default_registry, summarize
from coola.testing.marks import (
    jax_available,
    numpy_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    pydantic_available,
    torch_available,
    xarray_available,
)
from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_pydantic_available,
    is_torch_available,
    is_xarray_available,
)
from tests.helpers import nest
from tests.integration.helpers import run_threads

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
if is_pydantic_available():
    from pydantic import BaseModel, SecretStr
if is_torch_available():
    import torch
if is_xarray_available():
    import xarray as xr


@dataclass
class Holder:
    items: list


##################################
#     Tests for threads          #
##################################


def test_summarize_thread_safe() -> None:
    """Test that concurrent calls to summarize() on the default registry
    give the same summary."""
    data = {"a": [1, 2, {3, 1, 2}], "b": {"c": "x\ny"}}
    expected = summarize(data, max_depth=4)
    results: list[str] = []
    lock = threading.Lock()

    def _worker() -> None:
        result = summarize(data, max_depth=4)
        with lock:
            results.append(result)

    run_threads([threading.Thread(target=_worker) for _ in range(32)])

    assert results == [expected] * 32


def test_summarize_thread_safe_with_a_shared_config() -> None:
    """Test that a config and a registry can be shared by threads."""
    registry = get_default_registry()
    config = SummaryConfig(max_depth=3, max_items=2)
    results: list[str] = []
    lock = threading.Lock()

    def _worker(index: int) -> None:
        result = registry.summarize(list(range(index + 3)), config)
        with lock:
            results.append(result)

    run_threads([threading.Thread(target=_worker, args=(i,)) for i in range(32)])

    assert len(results) == 32
    assert len(set(results)) == 32


####################################
#     Tests for the determinism    #
####################################


def _summary_of_a_set_of_strings(hash_seed: str) -> str:
    src = str(Path(coola.__file__).parents[1])
    code = (
        "from coola.summary import summarize; "
        "print(repr(summarize({'beta', 'alpha', 'gamma', 'delta', 'epsilon', 'zeta'}, max_items=3)))"
    )
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": src, "PYTHONHASHSEED": hash_seed},
    ).stdout


def test_summarize_set_does_not_depend_on_pythonhashseed() -> None:
    """Test that the summary of a set of strings is the same in two
    processes with a different hash seed."""
    assert _summary_of_a_set_of_strings("0") == _summary_of_a_set_of_strings("1")


def test_summarize_set_shows_the_smallest_items_in_order() -> None:
    assert _summary_of_a_set_of_strings("0") == (
        "\"<class 'set'> (length=6)\\n  (0): alpha\\n  (1): beta\\n  (2): delta\\n  ... (+3 more)\"\n"
    )


################################
#     Tests for the depth      #
################################


def test_summarize_very_deep_structure() -> None:
    """Test that a nesting well above the recursion limit does not
    raise."""
    assert summarize(nest(1, 20_000), max_depth=10**6, max_items=1).count("\n") == 20_000


def test_summarize_huge_collections_at_the_display_depth() -> None:
    assert summarize([list(range(1_000_000))] * 3) == (
        "<class 'list'> (length=3)\n"
        "  (0): [0, 1, 2, 3, 4, ...]\n"
        "  (1): [0, 1, 2, 3, 4, ...]\n"
        "  (2): [0, 1, 2, 3, 4, ...]"
    )


def test_summarize_dataclass_cycle() -> None:
    data = Holder([])
    data.items.append(data)
    assert "<cycle>" in summarize(data, max_depth=10)


###############################################
#     Tests with the optional libraries       #
###############################################


def _last_line(data: object) -> str:
    return summarize({"x": data}, max_depth=2).splitlines()[-1]


@numpy_available
def test_summarize_numpy_in_a_nested_structure() -> None:
    assert _last_line(np.ones((2, 3))).endswith("| shape=(2, 3) | dtype=float64")


@torch_available
def test_summarize_torch_in_a_nested_structure() -> None:
    assert _last_line(torch.ones(2)).endswith(
        "| shape=torch.Size([2]) | dtype=torch.float32 | device=cpu | requires_grad=False"
    )


@pandas_available
def test_summarize_pandas_in_a_nested_structure() -> None:
    assert _last_line(pd.DataFrame({"a": [1]})).endswith("| shape=(1, 1) | columns=a: int64")


@polars_available
def test_summarize_polars_in_a_nested_structure() -> None:
    assert _last_line(pl.DataFrame({"a": [1]})).endswith("| shape=(1, 1) | columns=a: Int64")


@polars_available
def test_summarize_polars_lazy_frame_in_a_nested_structure() -> None:
    assert _last_line(pl.LazyFrame({"a": [1]})).endswith("| columns=a: Int64")


@pyarrow_available
def test_summarize_pyarrow_in_a_nested_structure() -> None:
    assert _last_line(pa.table({"a": [1]})).endswith("| shape=(1, 1) | columns=a: int64")


@xarray_available
@numpy_available
def test_summarize_xarray_in_a_nested_structure() -> None:
    assert _last_line(xr.DataArray(np.ones(2), name="x")).endswith(
        "| dims=('dim_0',) | shape=(2,) | dtype=float64 | name='x'"
    )


@jax_available
def test_summarize_jax_in_a_nested_structure() -> None:
    assert _last_line(jnp.ones(2)).endswith("| shape=(2,) | dtype=float32")


@pydantic_available
def test_summarize_pydantic_model_hides_the_secrets() -> None:
    class Model(BaseModel):
        name: str
        password: SecretStr

    assert summarize(Model.model_validate({"name": "a", "password": "b"})) == (
        f"{Model} (fields=1)\n  (name): a"
    )


@polars_available
def test_summarize_lazy_frame_is_not_collected() -> None:
    frame = pl.LazyFrame({"a": [1, 2]}).with_columns(pl.col("a").map_elements(_fail))
    assert summarize(frame, max_depth=1).startswith("<class 'polars.lazyframe")


def _fail(value: int) -> int:  # noqa: ARG001
    msg = "the lazy frame was collected"
    raise RuntimeError(msg)


@numpy_available
def test_summarize_show_data_with_numpy() -> None:
    assert summarize({"x": np.arange(3)}, show_data=True) == (
        "<class 'dict'> (length=1)\n  (x): array([0, 1, 2])"
    )


@pandas_available
def test_summarize_max_items_limits_the_columns() -> None:
    frame = pd.DataFrame({f"c{i}": [i] for i in range(4)})
    assert summarize(frame, max_items=2).endswith("columns=c0: int64, c1: int64, ... (+2 more)")


@pytest.mark.parametrize("max_depth", [0, 1, 2])
def test_summarize_unprintable_object_does_not_raise(max_depth: int) -> None:
    class Unprintable:
        def __str__(self) -> str:
            msg = "no str"
            raise RuntimeError(msg)

        def __repr__(self) -> str:
            msg = "no repr"
            raise RuntimeError(msg)

    assert "Unprintable object at 0x" in summarize([Unprintable()], max_depth=max_depth)


#########################################
#     Contract of the summarizers       #
#########################################


def _samples() -> dict[str, object]:
    samples: dict[str, object] = {
        "object": object(),
        "int": 1,
        "str": "abc",
        "list": [1, 2],
        "tuple": (1, 2),
        "deque": deque([1, 2]),
        "set": {1, 2},
        "frozenset": frozenset({1, 2}),
        "dict": {"a": 1},
        "mapping_proxy": MappingProxyType({"a": 1}),
        "user_string": UserString("abc"),
        "dataclass": Holder([1]),
    }
    if is_numpy_available():
        samples["numpy"] = np.ones(2)
    if is_torch_available():
        samples["torch"] = torch.ones(2)
    if is_jax_available():
        samples["jax"] = jnp.ones(2)
    if is_pandas_available():
        samples["pandas_frame"] = pd.DataFrame({"a": [1]})
        samples["pandas_series"] = pd.Series([1])
    if is_polars_available():
        samples["polars_frame"] = pl.DataFrame({"a": [1]})
        samples["polars_lazy"] = pl.LazyFrame({"a": [1]})
        samples["polars_series"] = pl.Series([1])
    if is_pyarrow_available():
        samples["pyarrow_table"] = pa.table({"a": [1]})
        samples["pyarrow_array"] = pa.array([1])
    if is_xarray_available():
        samples["xarray_dataset"] = xr.Dataset({"a": ("x", [1])})
        samples["xarray_variable"] = xr.Variable("x", [1])
    return samples


@pytest.mark.parametrize("name", list(_samples()))
@pytest.mark.parametrize("compact", [True, False])
def test_default_summarizers_return_a_str_or_a_node(name: str, compact: bool) -> None:
    """Test that the summarizer of every default type returns a str or a
    Node, whose children are lazy."""
    data = _samples()[name]
    result = (
        get_default_registry()
        .find_summarizer(type(data))
        .summarize(data, SummaryConfig(), compact=compact)
    )
    assert isinstance(result, (str, Node))


@pytest.mark.parametrize("name", list(_samples()))
def test_default_summarizers_node_has_a_total_and_lazy_children(name: str) -> None:
    data = _samples()[name]
    result = (
        get_default_registry()
        .find_summarizer(type(data))
        .summarize(data, SummaryConfig(), compact=False)
    )
    assert isinstance(result, str) or (result.total >= 0 and not isinstance(result.children, list))

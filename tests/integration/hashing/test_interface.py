from __future__ import annotations

import threading
from typing import Any

import pytest

from coola.hashing import hash_object
from coola.identifier import generate_stable_content_id, generate_stable_uuid5
from coola.testing.marks import (
    numpy_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    pydantic_available,
    torch_available,
    xarray_available,
)
from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_pydantic_available,
    is_torch_available,
    is_xarray_available,
)
from tests.integration.helpers import run_python, run_threads

if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_pyarrow_available():
    import pyarrow as pa
if is_pydantic_available():
    import pydantic
if is_torch_available():
    import torch
if is_xarray_available():
    import xarray as xr

DATA = {"a": [1, 2, {"b": (3, 4)}], "c": {5, 6}, "d": "text"}


#################################
#     Tests for hash_object     #
#################################


def test_hash_object_thread_safe() -> None:
    """Test that concurrent calls to hash_object() on the default
    registry give the same hash."""
    results: list[str] = []
    lock = threading.Lock()

    def _worker() -> None:
        value = hash_object(DATA)
        with lock:
            results.append(value)

    run_threads([threading.Thread(target=_worker) for _ in range(16)])

    assert len(results) == 16
    assert len(set(results)) == 1


def test_hash_object_reentrant_threads_with_cycles_do_not_interfere() -> None:
    """Test that a thread that hashes a cyclic structure does not make
    the other threads report a cycle."""
    errors: list[BaseException] = []
    results: list[str] = []
    lock = threading.Lock()
    cyclic: list[Any] = []
    cyclic.append(cyclic)

    def _worker(index: int) -> None:
        try:
            value = hash_object(cyclic if index % 2 else DATA)
        except ValueError as exc:
            with lock:
                errors.append(exc)
        else:
            with lock:
                results.append(value)

    run_threads([threading.Thread(target=_worker, args=(i,)) for i in range(16)])

    assert len(errors) == 8
    assert len(set(results)) == 1


def test_hash_object_depth_of_one_hundred_thousand() -> None:
    data: list[Any] = []
    for _ in range(100_000):
        data = [data]
    assert len(hash_object(data)) == 64


def test_hash_object_dag_with_sixty_levels() -> None:
    data: list[Any] = []
    for _ in range(60):
        data = [data, data]
    assert len(hash_object(data)) == 64


def test_hash_object_is_independent_of_pythonhashseed() -> None:
    code = (
        "from coola.hashing import hash_object; "
        "print(hash_object({'a': {1, 2, 3}, 'b': frozenset({'x', 'y'}), 'c': [1.5, None]}))"
    )
    first = run_python(code, env={"PYTHONHASHSEED": "1"})
    second = run_python(code, env={"PYTHONHASHSEED": "2"})
    assert first == second


def test_stable_content_id_end_to_end() -> None:
    assert generate_stable_content_id(DATA) == generate_stable_content_id(DATA)


def test_stable_uuid5_end_to_end() -> None:
    assert generate_stable_uuid5(DATA) == generate_stable_uuid5(DATA)


########################################
#     Tests for the optional libraries     #
########################################


@numpy_available
def test_hash_object_numpy_array() -> None:
    assert hash_object(np.arange(4)) == hash_object(np.arange(4))


@torch_available
def test_hash_object_torch_tensor() -> None:
    assert hash_object(torch.arange(4)) == hash_object(torch.arange(4))


@pandas_available
def test_hash_object_pandas_dataframe_with_nested_values() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    assert hash_object({"frame": frame}) == hash_object({"frame": frame.copy()})


@polars_available
def test_hash_object_polars_dataframe() -> None:
    assert hash_object(pl.DataFrame({"a": [1, 2]})) == hash_object(pl.DataFrame({"a": [1, 2]}))


@pyarrow_available
def test_hash_object_pyarrow_table() -> None:
    assert hash_object(pa.table({"a": [1, 2]})) == hash_object(pa.table({"a": [1, 2]}))


@xarray_available
def test_hash_object_xarray_data_array() -> None:
    array = xr.DataArray([1, 2], dims=["x"], coords={"x": [0, 1]})
    assert hash_object(array) == hash_object(array.copy())


@pydantic_available
def test_hash_object_pydantic_model_nested_in_a_list() -> None:
    model = pydantic.create_model("Model", x=(int, ...))
    assert hash_object([model.model_validate({"x": 1})]) == hash_object(
        [model.model_validate({"x": 1})]
    )


@pydantic_available
def test_hash_object_pydantic_model_that_contains_itself_raises() -> None:
    class Node(pydantic.BaseModel):
        children: list[Any]

    node = Node(children=[])
    node.children.append(node)
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(node)

r"""Test the contract of ``encode`` for one sample per type of the
default registry."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest

from coola.hashing import Node, get_default_registry, hash_object
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
from tests.helpers import CONTAINER_SAMPLES, LEAF_SAMPLES

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


class Color(Enum):
    RED = 1


@dataclass
class Point:
    x: int
    y: str


class Content:
    def __coola_hash__(self) -> object:
        return (1, 2)


SAMPLES = [
    *LEAF_SAMPLES,
    *CONTAINER_SAMPLES,
    pytest.param(None, id="none"),
    pytest.param(b"abc", id="bytes"),
    pytest.param(bytearray(b"abc"), id="bytearray"),
    pytest.param(memoryview(b"abc"), id="memoryview"),
    pytest.param(Color.RED, id="enum"),
    pytest.param(OrderedDict(a=1, b=2), id="ordered_dict"),
    pytest.param(date(2024, 1, 2), id="date"),
    pytest.param(datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC), id="datetime"),
    pytest.param(time(3, 4, 5), id="time"),
    pytest.param(timedelta(days=1), id="timedelta"),
    pytest.param(timezone(timedelta(hours=1)), id="timezone"),
    pytest.param(UUID(int=1), id="uuid"),
    pytest.param(ZoneInfo("UTC"), id="zoneinfo"),
    pytest.param(Decimal("1.50"), id="decimal"),
    pytest.param(Fraction(1, 3), id="fraction"),
    pytest.param(range(5), id="range"),
    pytest.param(PurePosixPath("a/b"), id="path"),
    pytest.param(Point(1, "a"), id="dataclass"),
    pytest.param(Content(), id="content"),
]

LIBRARY_SAMPLES = [
    pytest.param(
        lambda: np.arange(6, dtype=np.float64).reshape(2, 3),
        marks=numpy_available,
        id="numpy_array",
    ),
    pytest.param(lambda: np.float32(1.5), marks=numpy_available, id="numpy_scalar"),
    pytest.param(lambda: torch.arange(4), marks=torch_available, id="torch"),
    pytest.param(lambda: pd.Series([1, 2], name="a"), marks=pandas_available, id="pandas_series"),
    pytest.param(
        lambda: pd.DataFrame({"a": [1, 2]}), marks=pandas_available, id="pandas_dataframe"
    ),
    pytest.param(lambda: pd.NaT, marks=pandas_available, id="pandas_nat"),
    pytest.param(lambda: pd.Period("2020"), marks=pandas_available, id="pandas_period"),
    pytest.param(lambda: pd.Interval(0, 1), marks=pandas_available, id="pandas_interval"),
    pytest.param(lambda: pl.Series("a", [1, 2]), marks=polars_available, id="polars_series"),
    pytest.param(
        lambda: pl.DataFrame({"a": [1, 2]}), marks=polars_available, id="polars_dataframe"
    ),
    pytest.param(lambda: pa.array([1, 2]), marks=pyarrow_available, id="pyarrow_array"),
    pytest.param(lambda: pa.table({"a": [1, 2]}), marks=pyarrow_available, id="pyarrow_table"),
    pytest.param(
        lambda: xr.DataArray([1, 2], dims=["x"]), marks=xarray_available, id="xarray_data_array"
    ),
    pytest.param(
        lambda: xr.Dataset({"a": ("x", [1, 2])}), marks=xarray_available, id="xarray_dataset"
    ),
    pytest.param(
        lambda: pydantic.create_model("Model", x=(int, ...)).model_validate({"x": 1}),
        marks=pydantic_available,
        id="pydantic",
    ),
]


def encode(sample: Any) -> bytes | Node:
    return get_default_registry().find_hasher(type(sample)).encode(sample)


def describe(encoding: bytes | Node) -> tuple[Any, ...]:
    if isinstance(encoding, bytes):
        return (encoding,)
    return (encoding.tag, encoding.ordered, encoding.group, len(list(encoding.children)))


@pytest.mark.parametrize("sample", SAMPLES)
def test_encode_returns_bytes_or_a_node(sample: Any) -> None:
    assert isinstance(encode(sample), bytes | Node)


@pytest.mark.parametrize("sample", SAMPLES)
def test_encode_is_the_same_twice(sample: Any) -> None:
    assert describe(encode(sample)) == describe(encode(sample))


@pytest.mark.parametrize("sample", SAMPLES)
def test_encode_does_not_modify_the_sample(sample: Any) -> None:
    before = repr(sample)
    encoding = encode(sample)
    if isinstance(encoding, Node):
        list(encoding.children)
    assert repr(sample) == before


@pytest.mark.parametrize("sample", SAMPLES)
def test_node_has_a_tag_and_a_valid_group(sample: Any) -> None:
    encoding = encode(sample)
    assert isinstance(encoding, bytes) or (encoding.tag != "" and encoding.group >= 1)


@pytest.mark.parametrize("sample", SAMPLES)
def test_hash_of_the_sample_is_deterministic(sample: Any) -> None:
    assert hash_object(sample) == hash_object(sample)


@pytest.mark.parametrize("factory", LIBRARY_SAMPLES)
def test_library_encode_returns_bytes_or_a_node(factory: Any) -> None:
    assert isinstance(encode(factory()), bytes | Node)


@pytest.mark.parametrize("factory", LIBRARY_SAMPLES)
def test_library_encode_is_the_same_twice(factory: Any) -> None:
    sample = factory()
    assert describe(encode(sample)) == describe(encode(sample))


@pytest.mark.parametrize("factory", LIBRARY_SAMPLES)
def test_library_node_has_a_tag_and_a_valid_group(factory: Any) -> None:
    encoding = encode(factory())
    assert isinstance(encoding, bytes) or (encoding.tag != "" and encoding.group >= 1)


@pytest.mark.parametrize("factory", LIBRARY_SAMPLES)
def test_library_hash_is_deterministic(factory: Any) -> None:
    assert hash_object(factory()) == hash_object(factory())

r"""Check that ``equality``, ``hashing`` and ``summary`` support the same
types.

Each row is a type with a factory that builds equal objects. The three
modules are called on two objects built by the factory. The known gaps
are listed in ``HASH_GAPS`` and are ``xfail(strict=True)``: fixing a gap
makes the test fail until the type is removed from the list, and a new
type must be added to ``TYPES``.
"""

from __future__ import annotations

import array
import collections
import dataclasses
import datetime
import decimal
import enum
import fractions
import pathlib
import re
import types
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import objects_are_equal
from coola.hashing import hash_object
from coola.summary import summarize
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

if TYPE_CHECKING:
    from collections.abc import Callable, Set as AbstractSet


class Color(enum.Enum):
    RED = 1


@dataclasses.dataclass
class Point:
    x: int


def builtin_types() -> dict[str, Callable[[], Any]]:
    mapping = {"a": 1}
    return {
        "int": lambda: 1,
        "float": lambda: 1.5,
        "complex": lambda: 1 + 2j,
        "bool": lambda: True,
        "None": lambda: None,
        "str": lambda: "a",
        "bytes": lambda: b"a",
        "bytearray": lambda: bytearray(b"a"),
        "list": lambda: [1],
        "tuple": lambda: (1,),
        "deque": lambda: collections.deque([1]),
        "dict": lambda: dict(mapping),
        "OrderedDict": lambda: collections.OrderedDict(mapping),
        "Counter": lambda: collections.Counter("aab"),
        "defaultdict": lambda: collections.defaultdict(int, mapping),
        "set": lambda: {1},
        "frozenset": lambda: frozenset({1}),
        "range": lambda: range(3),
        "slice": lambda: slice(1, 2),
        "Ellipsis": lambda: ...,
        "dict_keys": lambda: dict(mapping).keys(),
        "dict_values": lambda: dict(mapping).values(),
        "dict_items": lambda: dict(mapping).items(),
        "mappingproxy": lambda: types.MappingProxyType(dict(mapping)),
        "array.array": lambda: array.array("i", [1]),
        "Decimal": lambda: decimal.Decimal("1.5"),
        "Fraction": lambda: fractions.Fraction(1, 3),
        "date": lambda: datetime.date(2020, 1, 1),
        "datetime": lambda: datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC),
        "time": lambda: datetime.time(1),
        "timedelta": lambda: datetime.timedelta(1),
        "timezone": lambda: datetime.timezone(datetime.timedelta(hours=1)),
        "PurePath": lambda: pathlib.PurePath("a"),
        "UUID": lambda: uuid.UUID(int=1),
        "Enum": lambda: Color.RED,
        "dataclass": lambda: Point(1),
        "Pattern": lambda: re.compile("a"),
    }


def numpy_types() -> dict[str, Callable[[], Any]]:
    return {
        "np.ndarray": lambda: np.arange(3),
        "np.float64": lambda: np.float64(1),
        "np.int64": lambda: np.int64(1),
        "np.bool_": lambda: np.bool_(True),
        "np.datetime64": lambda: np.datetime64("2020-01-01"),
        "np.timedelta64": lambda: np.timedelta64(1, "D"),
        "np.void": lambda: np.zeros(1, dtype=[("a", "i4")])[0],
        "np.structured": lambda: np.zeros(2, dtype=[("a", "i4")]),
        "np.ma.MaskedArray": lambda: np.ma.array([1, 2], mask=[0, 1]),
    }


def pandas_types() -> dict[str, Callable[[], Any]]:
    return {
        "pd.Series": lambda: pd.Series([1]),
        "pd.DataFrame": lambda: pd.DataFrame({"a": [1]}),
        "pd.Index": lambda: pd.Index([1]),
        "pd.MultiIndex": lambda: pd.MultiIndex.from_tuples([(1, 2)]),
        "pd.DatetimeIndex": lambda: pd.DatetimeIndex(["2020-01-01"]),
        "pd.RangeIndex": lambda: pd.RangeIndex(3),
        "pd.Categorical": lambda: pd.Categorical(["a"]),
        "pd.NA": lambda: pd.NA,
        "pd.NaT": lambda: pd.NaT,
        "pd.Timestamp": lambda: pd.Timestamp("2020-01-01"),
        "pd.Interval": lambda: pd.Interval(0, 1),
        "pd.Period": lambda: pd.Period("2020"),
    }


def polars_types() -> dict[str, Callable[[], Any]]:
    return {
        "pl.Series": lambda: pl.Series([1]),
        "pl.DataFrame": lambda: pl.DataFrame({"a": [1]}),
        "pl.LazyFrame": lambda: pl.LazyFrame({"a": [1]}),
    }


def pyarrow_types() -> dict[str, Callable[[], Any]]:
    return {
        "pa.Array": lambda: pa.array([1]),
        "pa.Table": lambda: pa.table({"a": [1]}),
        "pa.Schema": lambda: pa.schema([("a", pa.int64())]),
        "pa.DataType": pa.int64,
        "pa.Scalar": lambda: pa.scalar(1),
    }


def torch_types() -> dict[str, Callable[[], Any]]:
    return {
        "torch.Tensor": lambda: torch.arange(3),
        "torch.dtype": lambda: torch.float32,
        "torch.device": lambda: torch.device("cpu"),
        "torch.meta": lambda: torch.empty(2, device="meta"),
        "torch.sparse": lambda: torch.eye(2).to_sparse(),
    }


def jax_types() -> dict[str, Callable[[], Any]]:
    return {"jax.Array": lambda: jnp.arange(3)}


def xarray_types() -> dict[str, Callable[[], Any]]:
    return {
        "xr.DataArray": lambda: xr.DataArray([1], dims=["x"]),
        "xr.Dataset": lambda: xr.Dataset({"a": ("x", [1])}),
        "xr.Variable": lambda: xr.Variable("x", [1]),
    }


TYPES: dict[str, Callable[[], Any]] = builtin_types()
for available, getter in [
    (is_numpy_available(), numpy_types),
    (is_pandas_available(), pandas_types),
    (is_polars_available(), polars_types),
    (is_pyarrow_available(), pyarrow_types),
    (is_torch_available(), torch_types),
    (is_jax_available(), jax_types),
    (is_xarray_available(), xarray_types),
]:
    if available:
        TYPES.update(getter())

# The types that equality and summary support, but hashing does not (14.4, 14.5).
HASH_GAPS = {
    "slice",
    "Ellipsis",
    "dict_keys",
    "dict_values",
    "dict_items",
    "array.array",
    "Pattern",
    "np.datetime64",
    "np.timedelta64",
    "np.void",
    "pd.Index",
    "pd.MultiIndex",
    "pd.DatetimeIndex",
    "pd.RangeIndex",
    "pd.Categorical",
    "pl.LazyFrame",
    "pa.Schema",
    "pa.DataType",
    "pa.Scalar",
    "torch.dtype",
    "torch.device",
    "torch.meta",
}


def params(gaps: AbstractSet[str] = frozenset()) -> list[Any]:
    return [
        pytest.param(
            factory,
            id=name,
            marks=pytest.mark.xfail(strict=True, reason="known gap") if name in gaps else (),
        )
        for name, factory in TYPES.items()
    ]


@pytest.mark.parametrize("factory", params())
def test_support_matrix_equal(factory: Callable[[], Any]) -> None:
    assert objects_are_equal(factory(), factory())


@pytest.mark.parametrize("factory", params(HASH_GAPS))
def test_support_matrix_hash(factory: Callable[[], Any]) -> None:
    assert hash_object(factory()) == hash_object(factory())


@pytest.mark.parametrize("factory", params())
def test_support_matrix_summarize(factory: Callable[[], Any]) -> None:
    assert isinstance(summarize(factory()), str)

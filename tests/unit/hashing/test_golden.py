r"""Golden vectors: one fixed hash per hasher family.

A failure of these tests means that the hash format changed. Update a
value only for a deliberate change of the format, and say so in the
changelog: the stable identifiers (``coola.identifier``) change with it.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

import pytest

from coola.hashing import hash_object
from coola.testing.fixtures import (
    numpy_available,
    pandas_available,
    polars_available,
    pyarrow_available,
    torch_available,
)
from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
)

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


class Color(Enum):
    RED = 1


@dataclass
class Point:
    x: int
    y: str


#################################
#     Tests for the builtins    #
#################################


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param("hello", "2998e64e4b20ac91", id="str"),
        pytest.param(b"hello", "c0849551a6d5dac1", id="bytes"),
        pytest.param(42, "ebaaf8a4ccc89421", id="int"),
        pytest.param(1.5, "5ae23551bd86d7cd", id="float"),
        pytest.param(True, "b125316ff709ad95", id="bool"),
        pytest.param(None, "f8a2900c635ef9c7", id="none"),
        pytest.param(1 + 2j, "475b70f206b62fc9", id="complex"),
        pytest.param([1, 2, 3], "824284b9d2ee9802", id="list"),
        pytest.param((1, 2, 3), "039d1e5e77772efd", id="tuple"),
        pytest.param(deque([1, 2]), "bb65e1fb574a17c8", id="deque"),
        pytest.param({"a": 1, "b": [1, 2]}, "fb93c37a6ee4027c", id="dict"),
        pytest.param(OrderedDict(a=1, b=2), "35f6edcaea773cea", id="ordered_dict"),
        pytest.param({1, 2, 3}, "4f167270019d90eb", id="set"),
        pytest.param(frozenset({"a", "b"}), "0440abede833eb60", id="frozenset"),
        pytest.param(date(2020, 1, 2), "8c1f8affa3b396e6", id="date"),
        pytest.param(datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC), "ef744648301e4d3a", id="datetime"),
        pytest.param(time(12, 30), "171974b74f846892", id="time"),
        pytest.param(timedelta(days=1), "5bd56b8d9aa13658", id="timedelta"),
        pytest.param(Color.RED, "36e9e174b6efc6e9", id="enum"),
        pytest.param(PurePosixPath("a/b"), "7fae459e4cdcf0a3", id="path"),
        pytest.param(UUID(int=1), "c967ba56faef28ea", id="uuid"),
        pytest.param(Decimal("1.5"), "3650360fd2ef1106", id="decimal"),
        pytest.param(Fraction(1, 3), "69572b36ed294f9a", id="fraction"),
        pytest.param(range(3), "ce60d9991fc858cf", id="range"),
        pytest.param(Point(1, "a"), "3655ae9ace425548", id="dataclass"),
    ],
)
def test_golden_vectors(data: Any, expected: str) -> None:
    assert hash_object(data, length=16) == expected


#####################################
#     Tests for the libraries       #
#####################################


@numpy_available
def test_golden_vector_numpy() -> None:
    assert hash_object(np.arange(6, dtype=np.int64).reshape(2, 3), length=16) == "11be6c42239ff9ff"


@pandas_available
def test_golden_vector_pandas_series() -> None:
    assert hash_object(pd.Series([1, 2], name="a"), length=16) == "e3ca5082f10d22a4"


@pandas_available
def test_golden_vector_pandas_dataframe() -> None:
    assert hash_object(pd.DataFrame({"a": [1, 2]}), length=16) == "73bcd65362dc4abc"


@polars_available
def test_golden_vector_polars_series() -> None:
    assert hash_object(pl.Series("a", [1, 2]), length=16) == "db7c9564e0871eae"


@pyarrow_available
def test_golden_vector_pyarrow_array() -> None:
    assert hash_object(pa.array([1, 2]), length=16) == "b091529d4b4a53ec"


@torch_available
def test_golden_vector_torch_tensor() -> None:
    assert hash_object(torch.arange(4, dtype=torch.float32), length=16) == "1c09475c78a6bc15"

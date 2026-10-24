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
        pytest.param("hello", "8145498007a9ef19", id="str"),
        pytest.param(b"hello", "c03fc993641ae772", id="bytes"),
        pytest.param(42, "4d693658248572a8", id="int"),
        pytest.param(1.5, "42d0502eb37c390c", id="float"),
        pytest.param(True, "fa7f7697534a7d69", id="bool"),
        pytest.param(None, "cb6d68ff841e0a60", id="none"),
        pytest.param(1 + 2j, "39d936881d8c05bd", id="complex"),
        pytest.param([1, 2, 3], "12a6b7d5826a7288", id="list"),
        pytest.param((1, 2, 3), "34e154b7f4f4cb11", id="tuple"),
        pytest.param(deque([1, 2]), "c2feb3f4e0f0bc75", id="deque"),
        pytest.param({"a": 1, "b": [1, 2]}, "4277a5be63f0836c", id="dict"),
        pytest.param(OrderedDict(a=1, b=2), "288b99159f272e4c", id="ordered_dict"),
        pytest.param({1, 2, 3}, "6b77c5c76e2ddefa", id="set"),
        pytest.param(frozenset({"a", "b"}), "c37951d9d4b066dd", id="frozenset"),
        pytest.param(date(2020, 1, 2), "4f441beb24e62e08", id="date"),
        pytest.param(datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC), "5f5247f938fb6eb5", id="datetime"),
        pytest.param(time(12, 30), "87148fd77014937f", id="time"),
        pytest.param(timedelta(days=1), "40c90371ff69e146", id="timedelta"),
        pytest.param(Color.RED, "24acae24d0bd6b1d", id="enum"),
        pytest.param(PurePosixPath("a/b"), "dfa2d76c35b9c767", id="path"),
        pytest.param(UUID(int=1), "870eac9d23e5da22", id="uuid"),
        pytest.param(Decimal("1.5"), "f42c1a9e187e3623", id="decimal"),
        pytest.param(Fraction(1, 3), "b23963a7df781152", id="fraction"),
        pytest.param(range(3), "111f090f7768723d", id="range"),
        pytest.param(Point(1, "a"), "37f8b40f6fd6a2de", id="dataclass"),
    ],
)
def test_golden_vectors(data: Any, expected: str) -> None:
    assert hash_object(data, length=16) == expected


#####################################
#     Tests for the libraries       #
#####################################


@numpy_available
def test_golden_vector_numpy() -> None:
    assert hash_object(np.arange(6, dtype=np.int64).reshape(2, 3), length=16) == "16b1596cd0904a5e"


@pandas_available
def test_golden_vector_pandas_series() -> None:
    assert hash_object(pd.Series([1, 2], name="a"), length=16) == "aeb5fa63a631eb96"


@pandas_available
def test_golden_vector_pandas_dataframe() -> None:
    assert hash_object(pd.DataFrame({"a": [1, 2]}), length=16) == "69458569da15a1f7"


@polars_available
def test_golden_vector_polars_series() -> None:
    assert hash_object(pl.Series("a", [1, 2]), length=16) == "1c892ca8442571af"


@pyarrow_available
def test_golden_vector_pyarrow_array() -> None:
    assert hash_object(pa.array([1, 2]), length=16) == "bf6f094668f367d0"


@torch_available
def test_golden_vector_torch_tensor() -> None:
    assert hash_object(torch.arange(4, dtype=torch.float32), length=16) == "5e64bab9594ae8f4"

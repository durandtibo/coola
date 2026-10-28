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
from coola.testing.marks import (
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
        pytest.param("hello", "0a465ca3e88a3a4c", id="str"),
        pytest.param(b"hello", "59f582b5108d6c1c", id="bytes"),
        pytest.param(42, "523ebc8accc11fcf", id="int"),
        pytest.param(1.5, "bb7d0092c5be7ab0", id="float"),
        pytest.param(True, "2dabca2e8bc28762", id="bool"),
        pytest.param(None, "1e04f01fed015b7c", id="none"),
        pytest.param(1 + 2j, "06abc3468c3b9ea9", id="complex"),
        pytest.param([1, 2, 3], "b783328a416de7dd", id="list"),
        pytest.param((1, 2, 3), "d942ea47e6839ee5", id="tuple"),
        pytest.param(deque([1, 2]), "b88729ee89f5836b", id="deque"),
        pytest.param({"a": 1, "b": [1, 2]}, "19c3aca4c41cf499", id="dict"),
        pytest.param(OrderedDict(a=1, b=2), "b18eb1f07613c5c0", id="ordered_dict"),
        pytest.param({1, 2, 3}, "3ba373d2e386945c", id="set"),
        pytest.param(frozenset({"a", "b"}), "e0004934794e6fc7", id="frozenset"),
        pytest.param(date(2020, 1, 2), "755d7c8e279de888", id="date"),
        pytest.param(datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC), "8084880dcf26cc8a", id="datetime"),
        pytest.param(time(12, 30), "87dbf4242693ae3d", id="time"),
        pytest.param(timedelta(days=1), "704975e7a2e57aa0", id="timedelta"),
        pytest.param(Color.RED, "73f359604b2faa89", id="enum"),
        pytest.param(PurePosixPath("a/b"), "c3e1743441d86c59", id="path"),
        pytest.param(UUID(int=1), "fe3c013a7e38167e", id="uuid"),
        pytest.param(Decimal("1.5"), "70943555cdb50467", id="decimal"),
        pytest.param(Fraction(1, 3), "2b7562b0d345efef", id="fraction"),
        pytest.param(range(3), "68bdbc27a0ad8142", id="range"),
        pytest.param(Point(1, "a"), "932bec13a7d3f950", id="dataclass"),
    ],
)
def test_golden_vectors(data: Any, expected: str) -> None:
    assert hash_object(data, length=16) == expected


#####################################
#     Tests for the libraries       #
#####################################


@numpy_available
def test_golden_vector_numpy() -> None:
    assert hash_object(np.arange(6, dtype=np.int64).reshape(2, 3), length=16) == "40143247a1021b65"


@pandas_available
def test_golden_vector_pandas_series() -> None:
    assert hash_object(pd.Series([1, 2], name="a"), length=16) == "1b20ced2f5504b26"


@pandas_available
def test_golden_vector_pandas_dataframe() -> None:
    assert hash_object(pd.DataFrame({"a": [1, 2]}), length=16) == "08143bc5fe7b5cab"


@polars_available
def test_golden_vector_polars_series() -> None:
    assert hash_object(pl.Series("a", [1, 2]), length=16) == "9504a6522469a81f"


@pyarrow_available
def test_golden_vector_pyarrow_array() -> None:
    assert hash_object(pa.array([1, 2]), length=16) == "95c31fa472871168"


@torch_available
def test_golden_vector_torch_tensor() -> None:
    assert hash_object(torch.arange(4, dtype=torch.float32), length=16) == "9bd680607be34c1d"

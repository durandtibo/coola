from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from coola.rounding import round_values
from coola.testing.marks import (
    numpy_available,
    pandas_available,
    polars_available,
    torch_available,
)
from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
)

if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd
if is_polars_available():
    import polars as pl
if is_torch_available():
    import torch

#####################################
#     Tests for round_values        #
#####################################


def test_round_values_default() -> None:
    assert round_values(2.567) == 3.0


def test_round_values_float() -> None:
    assert round_values(2.567, mode="floor", decimals=2) == 2.56


def test_round_values_float_type() -> None:
    assert type(round_values(2.5)) is float


def test_round_values_int() -> None:
    assert round_values(3, mode="ceil", decimals=2) == 3


@pytest.mark.parametrize(("mode", "expected"), [("round", 1.2), ("floor", 1.2), ("ceil", 1.3)])
def test_round_values_modes(mode: str, expected: float) -> None:
    assert round_values(1.234, mode=mode, decimals=1) == expected  # ty: ignore[invalid-argument-type]


def test_round_values_incorrect_mode() -> None:
    with pytest.raises(ValueError, match=r"mode must be one of"):
        round_values(1.5, mode="trunc")  # ty: ignore[invalid-argument-type]


def test_round_values_negative_decimals() -> None:
    with pytest.raises(ValueError, match=r"decimals must be greater than or equal to 0"):
        round_values(1.5, decimals=-1)


def test_round_values_decimals_not_int() -> None:
    with pytest.raises(TypeError, match=r"decimals must be an int"):
        round_values(1.5, decimals=1.0)  # ty: ignore[invalid-argument-type]


def test_round_values_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match=r"positional"):
        round_values(1.5, "floor")  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize("data", ["abc", None, Decimal("1.5"), [1.5], {"a": 1.5}, object()])
def test_round_values_unsupported(data: object) -> None:
    with pytest.raises(TypeError, match=r"Unsupported type"):
        round_values(data)


def test_round_values_recursive_nested() -> None:
    data = {"a": [1.234, "x", None], "b": (5.678, 3), "c": {2.5}}
    assert round_values(data, mode="floor", decimals=1, recursive=True) == {
        "a": [1.2, "x", None],
        "b": (5.6, 3),
        "c": {2.5},
    }


def test_round_values_recursive_does_not_modify_input() -> None:
    data = {"a": [1.234]}
    round_values(data, recursive=True)
    assert data == {"a": [1.234]}


def test_round_values_recursive_leaf() -> None:
    assert round_values(2.567, decimals=1, recursive=True) == 2.6


@dataclass
class Point:
    x: float


def test_round_values_recursive_enters_the_dataclass() -> None:
    assert round_values(Point(1.234), decimals=1, recursive=True) == Point(1.2)


@numpy_available
def test_round_values_numpy_array() -> None:
    assert round_values(np.array([1.234, 5.678]), mode="floor", decimals=1).tolist() == [1.2, 5.6]


@numpy_available
def test_round_values_numpy_float64_scalar() -> None:
    assert type(round_values(np.float64(1.234), decimals=1)) is np.float64


@numpy_available
def test_round_values_recursive_numpy() -> None:
    result = round_values([np.array([1.26])], mode="floor", decimals=1, recursive=True)
    assert result[0].tolist() == [1.2]


@torch_available
def test_round_values_torch_tensor() -> None:
    result = round_values(
        torch.tensor([1.234, 5.678], dtype=torch.float64), mode="ceil", decimals=1
    )
    assert result.tolist() == [1.3, 5.7]


@torch_available
def test_round_values_torch_tensor_subclass() -> None:
    result = round_values(torch.nn.Parameter(torch.tensor([1.5, 2.5])))
    assert result.tolist() == [2.0, 2.0]


@pandas_available
def test_round_values_pandas_dataframe() -> None:
    result = round_values(pd.DataFrame({"a": [1.234], "b": ["x"]}), decimals=1)
    assert result.to_dict("list") == {"a": [1.2], "b": ["x"]}


@polars_available
def test_round_values_polars_dataframe() -> None:
    result = round_values(pl.DataFrame({"a": [1.234], "b": ["x"]}), decimals=1)
    assert result.to_dict(as_series=False) == {"a": [1.2], "b": ["x"]}


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
@torch_available
@numpy_available
def test_round_values_numpy_and_torch_agree(mode: str) -> None:
    values = [0.29, 0.07, -2.567, 2.5, 1.005, 1e-3]
    expected = round_values(np.array(values), mode=mode, decimals=2).tolist()  # ty: ignore[invalid-argument-type]
    assert (
        round_values(torch.tensor(values, dtype=torch.float64), mode=mode, decimals=2).tolist()
        == expected
    )  # ty: ignore[invalid-argument-type]


@numpy_available
def test_round_values_numpy_integer_decimals() -> None:
    assert round_values(1.26, mode="floor", decimals=np.int64(1)) == 1.2

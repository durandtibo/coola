from __future__ import annotations

import math
from decimal import Decimal

import pytest

from coola.rounding.native import round_native

####################################
#     Tests for round_native       #
####################################


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
@pytest.mark.parametrize("value", [0, 3, -3, 2**100])
def test_round_native_int_unchanged(mode: str, value: int) -> None:
    assert round_native(value, mode, 2) == value  # ty: ignore[invalid-argument-type]


def test_round_native_int_type() -> None:
    assert type(round_native(3, "floor", 0)) is int


def test_round_native_bool_unchanged() -> None:
    assert round_native(True, "ceil", 0) is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2.5, 2.0), (3.5, 4.0), (-2.5, -2.0), (2.567, 3.0), (-2.567, -3.0), (0.4, 0.0)],
)
def test_round_native_round(value: float, expected: float) -> None:
    assert round_native(value, "round", 0) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(2.567, 2.0), (-2.567, -3.0), (2.0, 2.0), (-0.1, -1.0), (0.9, 0.0)]
)
def test_round_native_floor(value: float, expected: float) -> None:
    assert round_native(value, "floor", 0) == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(2.567, 3.0), (-2.567, -2.0), (2.0, 2.0), (0.1, 1.0), (-0.9, 0.0)]
)
def test_round_native_ceil(value: float, expected: float) -> None:
    assert round_native(value, "ceil", 0) == expected


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
def test_round_native_float_type(mode: str) -> None:
    assert type(round_native(2.5, mode, 0)) is float  # ty: ignore[invalid-argument-type]


def test_round_native_round_decimals() -> None:
    assert round_native(2.567, "round", 2) == 2.57


def test_round_native_floor_decimals() -> None:
    assert round_native(2.567, "floor", 2) == 2.56


def test_round_native_ceil_decimals() -> None:
    assert round_native(2.561, "ceil", 2) == 2.57


@pytest.mark.parametrize("value", [0.29, 0.57, 1.1, 4.35])
def test_round_native_floor_value_on_the_grid(value: float) -> None:
    assert round_native(value, "floor", 2) == value


@pytest.mark.parametrize("value", [0.07, 0.57, 1.1, 4.35])
def test_round_native_ceil_value_on_the_grid(value: float) -> None:
    assert round_native(value, "ceil", 2) == value


def test_round_native_round_halfway_exact_binary() -> None:
    assert round_native(2.675, "round", 2) == 2.67


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_round_native_infinite(mode: str, value: float) -> None:
    assert round_native(value, mode, 1) == value  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
def test_round_native_nan(mode: str) -> None:
    assert math.isnan(round_native(math.nan, mode, 1))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
def test_round_native_large_value_unchanged(mode: str) -> None:
    assert round_native(1e308, mode, 2) == 1e308  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
def test_round_native_value_larger_than_2_53_unchanged(mode: str) -> None:
    assert round_native(123456789.123, mode, 12) == 123456789.123  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
def test_round_native_too_many_decimals_unchanged(mode: str) -> None:
    assert round_native(0.1, mode, 400) == 0.1  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", ["1.5", None, Decimal("1.5"), [1.5], 1 + 2j])
def test_round_native_unsupported(value: object) -> None:
    with pytest.raises(TypeError, match=r"Unsupported type"):
        round_native(value, "round", 0)

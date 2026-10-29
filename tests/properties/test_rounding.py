r"""Property-based tests for the properties of ``round_values``.

The properties are documented in ``docs/docs/uguide/rounding.md``.
"""

from __future__ import annotations

import math

import pytest

from coola.rounding import round_values
from coola.testing.marks import numpy_available, polars_available, pyarrow_available
from coola.utils.imports import is_numpy_available, is_polars_available, is_pyarrow_available
from tests.unit.rounding.helpers import EXACT_BOUND, oracle

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import (  # noqa: E402
    example,
    given,
    strategies as st,
)

if is_numpy_available():
    import numpy as np
if is_polars_available():
    import polars as pl
if is_pyarrow_available():
    import pyarrow as pa

modes = st.sampled_from(["round", "floor", "ceil"])
decimals_ = st.integers(min_value=0, max_value=8)


@st.composite
def floats_near_grid(draw: st.DrawFn) -> tuple[float, int]:
    # A decimal with 1 to 17 significant digits, or a neighbour (one or two ulps) of a grid point
    # or of a halfway case, with the number of decimals to round to.
    decimals = draw(decimals_)
    if draw(st.booleans()):
        x = draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
    else:
        numerator = draw(st.integers(min_value=-(10**9), max_value=10**9))
        x = (numerator * 5) / 10 ** (decimals + 1)
        for _ in range(draw(st.integers(min_value=0, max_value=2))):
            x = math.nextafter(x, draw(st.sampled_from([-math.inf, math.inf])))
    return x, decimals


def _bits(value: float) -> tuple[float, float]:
    # ``==`` hides the sign of a zero.
    return value, math.copysign(1.0, value)


def _is_exact(x: float, decimals: int) -> bool:
    return abs(x) * 10.0**decimals < EXACT_BOUND


#############################
#     Against the oracle    #
#############################


@given(case=floats_near_grid(), mode=modes)
@example(case=(2.675, 2), mode="round")
@example(case=(5.5200000000000005, 2), mode="ceil")
def test_round_values_float_matches_oracle(case: tuple[float, int], mode: str) -> None:
    x, decimals = case
    if _is_exact(x, decimals):
        assert _bits(round_values(x, mode=mode, decimals=decimals)) == _bits(
            oracle(x, mode, decimals)
        )


@numpy_available
@given(case=floats_near_grid(), mode=modes)
@example(case=(5.5200000000000005, 2), mode="ceil")
def test_round_values_numpy_matches_oracle(case: tuple[float, int], mode: str) -> None:
    x, decimals = case
    if _is_exact(x, decimals):
        result = round_values(np.array([x]), mode=mode, decimals=decimals)
        assert _bits(result.item()) == _bits(oracle(x, mode, decimals))


####################
#     Contract     #
####################


@given(case=floats_near_grid())
@example(case=(5.5200000000000005, 2))
@example(case=(3.3000000000000003, 1))
def test_round_values_floor_is_below_and_ceil_is_above(case: tuple[float, int]) -> None:
    x, decimals = case
    low = round_values(x, mode="floor", decimals=decimals)
    high = round_values(x, mode="ceil", decimals=decimals)
    assert low <= x <= high


@given(case=floats_near_grid(), mode=modes)
def test_round_values_float_is_idempotent(case: tuple[float, int], mode: str) -> None:
    x, decimals = case
    once = round_values(x, mode=mode, decimals=decimals)
    assert _bits(round_values(once, mode=mode, decimals=decimals)) == _bits(once)


@given(case=floats_near_grid(), mode=modes, other=st.floats(min_value=0, max_value=1e-3))
def test_round_values_float_is_monotone(case: tuple[float, int], mode: str, other: float) -> None:
    x, decimals = case
    assert round_values(x, mode=mode, decimals=decimals) <= round_values(
        x + other, mode=mode, decimals=decimals
    )


@given(case=floats_near_grid())
def test_round_values_round_is_at_most_half_a_step_away(case: tuple[float, int]) -> None:
    x, decimals = case
    if _is_exact(x, decimals):
        step = 10.0**-decimals
        assert abs(round_values(x, decimals=decimals) - x) <= step / 2 + 2 * math.ulp(x)


@given(x=st.floats(min_value=-0.49, max_value=-0.0, allow_nan=False), mode=modes)
@example(x=-0.4, mode="ceil")
def test_round_values_float_keeps_the_sign_of_a_zero_result(x: float, mode: str) -> None:
    result = round_values(x, mode=mode)
    assert math.copysign(1.0, result) == -1.0 or result != 0.0


@given(x=st.floats(allow_nan=True), mode=modes)
def test_round_values_integer_valued_float_stays_a_float(x: float, mode: str) -> None:
    assert isinstance(round_values(x, mode=mode, decimals=2), float)


#######################
#     Same bits       #
#######################


@numpy_available
@given(case=floats_near_grid(), mode=modes)
@example(case=(2.675, 2), mode="round")
def test_round_values_float_and_numpy_give_the_same_bits(
    case: tuple[float, int], mode: str
) -> None:
    x, decimals = case
    scalar = round_values(x, mode=mode, decimals=decimals)
    array = round_values(np.array([x]), mode=mode, decimals=decimals).item()
    assert _bits(scalar) == _bits(array)


@polars_available
@numpy_available
@given(case=floats_near_grid(), mode=modes)
@example(case=(-4.822, 1), mode="round")
def test_round_values_polars_and_numpy_give_the_same_bits(
    case: tuple[float, int], mode: str
) -> None:
    x, decimals = case
    column = round_values(pl.Series([x, 12.0]), mode=mode, decimals=decimals).to_list()[0]
    array = round_values(np.array([x]), mode=mode, decimals=decimals).item()
    assert _bits(column) == _bits(array)


@pyarrow_available
@numpy_available
@given(case=floats_near_grid(), mode=modes)
@example(case=(0.225, 2), mode="round")
def test_round_values_pyarrow_and_numpy_give_the_same_bits(
    case: tuple[float, int], mode: str
) -> None:
    x, decimals = case
    column = round_values(pa.array([x]), mode=mode, decimals=decimals).to_pylist()[0]
    array = round_values(np.array([x]), mode=mode, decimals=decimals).item()
    assert _bits(column) == _bits(array)

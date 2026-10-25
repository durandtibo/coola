from __future__ import annotations

import numpy as np
import pytest

from coola.equality.handler.elementwise import elementwise_close

NAN = float("nan")
INF = float("inf")


def _close(
    x: list[float], y: list[float], *, atol: float = 0.0, rtol: float = 0.0, equal_nan: bool = False
) -> list[bool]:
    with np.errstate(invalid="ignore"):  # ``inf - inf``: the caller silences NumPy
        return elementwise_close(
            np.array(x), np.array(y), atol=atol, rtol=rtol, equal_nan=equal_nan, xp=np
        ).tolist()


def test_elementwise_close_each_element_uses_its_own_scale() -> None:
    # |100 - 101.005| = 1.005 <= 0.01 * 101.005 for both pairs, whatever the direction.
    assert _close([100.0, 101.005], [101.005, 100.0], rtol=0.01) == [True, True]


def test_elementwise_close_is_symmetric() -> None:
    x, y = [100.0, 101.005, 1.0], [101.005, 100.0, 3.0]
    assert _close(x, y, rtol=0.01) == _close(y, x, rtol=0.01)


def test_elementwise_close_atol_and_rtol_add_up() -> None:
    # |1 - 3| = 2 <= 0.5 + 0.5 * 3
    assert _close([1.0], [3.0], atol=0.5, rtol=0.5) == [True]


def test_elementwise_close_outside_the_tolerance() -> None:
    assert _close([1.0], [3.5], atol=0.5, rtol=0.5) == [False]


def test_elementwise_close_boundary_is_close() -> None:
    assert _close([1.0], [2.0], atol=1.0) == [True]


def test_elementwise_close_no_tolerance_requires_equality() -> None:
    assert _close([1.0, 2.0], [1.0, 2.5]) == [True, False]


def test_elementwise_close_equal_infinities() -> None:
    assert _close([INF, -INF], [INF, -INF], atol=1.0) == [True, True]


def test_elementwise_close_different_infinities() -> None:
    assert _close([INF, INF], [-INF, 1.0], atol=1e300) == [False, False]


def test_elementwise_close_nan_not_equal_nan() -> None:
    assert _close([NAN], [NAN], atol=1.0) == [False]


def test_elementwise_close_nan_equal_nan() -> None:
    assert _close([NAN], [NAN], atol=1.0, equal_nan=True) == [True]


def test_elementwise_close_nan_and_number_equal_nan() -> None:
    assert _close([NAN, 1.0], [1.0, NAN], atol=1.0, equal_nan=True) == [False, False]


def test_elementwise_close_empty() -> None:
    assert _close([], [], atol=1.0) == []


def test_elementwise_close_complex() -> None:
    result = elementwise_close(
        np.array([1 + 1j]), np.array([1.5 + 1j]), atol=0.6, rtol=0.0, equal_nan=False, xp=np
    )
    assert result.tolist() == [True]


@pytest.mark.parametrize("equal_nan", [True, False])
def test_elementwise_close_does_not_mutate_the_inputs(equal_nan: bool) -> None:
    x, y = np.array([1.0, NAN]), np.array([1.1, NAN])
    elementwise_close(x, y, atol=0.2, rtol=0.0, equal_nan=equal_nan, xp=np)
    assert np.array_equal(x, [1.0, NAN], equal_nan=True)

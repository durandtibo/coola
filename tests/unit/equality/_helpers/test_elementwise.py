from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from coola.equality._helpers.elementwise import elementwise_close
from coola.testing.marks import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


NAN = float("nan")
INF = float("inf")


def _close(
    x: list[float], y: list[float], *, atol: float = 0.0, rtol: float = 0.0, equal_nan: bool = False
) -> list[bool]:
    with np.errstate(invalid="ignore"):  # ``inf - inf``: the caller silences NumPy
        return elementwise_close(
            np.array(x), np.array(y), atol=atol, rtol=rtol, equal_nan=equal_nan, xp=np
        ).tolist()


@numpy_available
def test_elementwise_close_each_element_uses_its_own_scale() -> None:
    # |100 - 101.005| = 1.005 <= 0.01 * 101.005 for both pairs, whatever the direction.
    assert _close([100.0, 101.005], [101.005, 100.0], rtol=0.01) == [True, True]


@numpy_available
def test_elementwise_close_is_symmetric() -> None:
    x, y = [100.0, 101.005, 1.0], [101.005, 100.0, 3.0]
    assert _close(x, y, rtol=0.01) == _close(y, x, rtol=0.01)


@numpy_available
def test_elementwise_close_atol_and_rtol_add_up() -> None:
    # |1 - 3| = 2 <= 0.5 + 0.5 * 3
    assert _close([1.0], [3.0], atol=0.5, rtol=0.5) == [True]


@numpy_available
def test_elementwise_close_outside_the_tolerance() -> None:
    assert _close([1.0], [3.5], atol=0.5, rtol=0.5) == [False]


@numpy_available
def test_elementwise_close_boundary_is_close() -> None:
    assert _close([1.0], [2.0], atol=1.0) == [True]


@numpy_available
def test_elementwise_close_no_tolerance_requires_equality() -> None:
    assert _close([1.0, 2.0], [1.0, 2.5]) == [True, False]


@numpy_available
def test_elementwise_close_equal_infinities() -> None:
    assert _close([INF, -INF], [INF, -INF], atol=1.0) == [True, True]


@numpy_available
def test_elementwise_close_different_infinities() -> None:
    assert _close([INF, INF], [-INF, 1.0], atol=1e300) == [False, False]


@numpy_available
def test_elementwise_close_nan_not_equal_nan() -> None:
    assert _close([NAN], [NAN], atol=1.0) == [False]


@numpy_available
def test_elementwise_close_nan_equal_nan() -> None:
    assert _close([NAN], [NAN], atol=1.0, equal_nan=True) == [True]


@numpy_available
def test_elementwise_close_nan_and_number_equal_nan() -> None:
    assert _close([NAN, 1.0], [1.0, NAN], atol=1.0, equal_nan=True) == [False, False]


@numpy_available
def test_elementwise_close_empty() -> None:
    assert _close([], [], atol=1.0) == []


@numpy_available
def test_elementwise_close_complex() -> None:
    result = elementwise_close(
        np.array([1 + 1j]), np.array([1.5 + 1j]), atol=0.6, rtol=0.0, equal_nan=False, xp=np
    )
    assert result.tolist() == [True]


@numpy_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_elementwise_close_does_not_mutate_the_inputs(equal_nan: bool) -> None:
    x, y = np.array([1.0, NAN]), np.array([1.1, NAN])
    elementwise_close(x, y, atol=0.2, rtol=0.0, equal_nan=equal_nan, xp=np)
    assert np.array_equal(x, [1.0, NAN], equal_nan=True)


def _int_close(
    x: list[int], y: list[int], *, dtype: str = "int64", atol: float = 0.0, rtol: float = 0.0
) -> list[bool]:
    return elementwise_close(
        np.array(x, dtype=dtype),
        np.array(y, dtype=dtype),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        xp=np,
    ).tolist()


def _torch_int_close(
    x: list[int], y: list[int], *, dtype: Any = None, atol: float = 0.0, rtol: float = 0.0
) -> list[bool]:
    return elementwise_close(
        torch.tensor(x, dtype=dtype or torch.int64),
        torch.tensor(y, dtype=dtype or torch.int64),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
        xp=torch,
    ).tolist()


@numpy_available
def test_elementwise_close_integers_differ_by_one_above_2_53() -> None:
    assert _int_close([2**62], [2**62 + 1], atol=0.5) == [False]


@numpy_available
def test_elementwise_close_integers_differ_by_one_above_2_53_reversed() -> None:
    assert _int_close([2**62 + 1], [2**62], atol=0.5) == [False]


@numpy_available
def test_elementwise_close_integers_equal_above_2_53() -> None:
    assert _int_close([2**62 + 1], [2**62 + 1]) == [True]


@numpy_available
def test_elementwise_close_integers_within_atol() -> None:
    assert _int_close([3], [5], atol=2.0) == [True]


@numpy_available
def test_elementwise_close_integers_outside_atol() -> None:
    assert _int_close([3], [5], atol=1.9) == [False]


@numpy_available
def test_elementwise_close_integers_rtol_scales_with_the_largest() -> None:
    assert _int_close([100], [110], rtol=0.1) == [True]


@numpy_available
def test_elementwise_close_int64_difference_wraps_around() -> None:
    # The exact difference is 2**64 - 1: the int64 subtraction wraps to -1.
    assert _int_close([-(2**63)], [2**63 - 1], atol=1e19) == [False]


@numpy_available
def test_elementwise_close_int64_difference_wraps_around_within_atol() -> None:
    assert _int_close([-(2**63)], [2**63 - 1], atol=1e30) == [True]


@numpy_available
def test_elementwise_close_uint64_extremes_outside_atol() -> None:
    assert _int_close([0], [2**64 - 1], dtype="uint64", atol=1e19) == [False]


@numpy_available
def test_elementwise_close_uint64_extremes_within_atol() -> None:
    assert _int_close([0], [2**64 - 1], dtype="uint64", atol=2e19) == [True]


@numpy_available
def test_elementwise_close_uint64_differ_by_one_above_2_53() -> None:
    assert _int_close([2**64 - 1], [2**64 - 2], dtype="uint64", atol=0.5) == [False]


@numpy_available
def test_elementwise_close_uint8_does_not_wrap_around() -> None:
    assert _int_close([255], [0], dtype="uint8", atol=254.0) == [False]


@numpy_available
def test_elementwise_close_int8_does_not_wrap_around() -> None:
    assert _int_close([-128], [127], dtype="int8", atol=254.0) == [False]


@numpy_available
def test_elementwise_close_integers_is_symmetric() -> None:
    x, y = [2**62, 1, -(2**63)], [2**62 + 1, 3, 2**63 - 1]
    assert _int_close(x, y, atol=1.0, rtol=1e-3) == _int_close(y, x, atol=1.0, rtol=1e-3)


@numpy_available
def test_elementwise_close_integers_does_not_mutate_the_inputs() -> None:
    x, y = np.array([1, 2], dtype="int8"), np.array([3, 4], dtype="int8")
    elementwise_close(x, y, atol=5.0, rtol=0.0, equal_nan=False, xp=np)
    assert x.dtype == np.int8


@numpy_available
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_elementwise_close_small_float_does_not_overflow(dtype: type) -> None:
    # |60000 - (-60000)| = 120000 > 1.5 * 60000: it overflows ``float16`` and wraps around.
    result = elementwise_close(
        np.array([60000], dtype=dtype),
        np.array([-60000], dtype=dtype),
        atol=0.0,
        rtol=1.5,
        equal_nan=False,
        xp=np,
    )
    assert result.tolist() == [False]


@numpy_available
def test_elementwise_close_float32_tolerance_in_double_precision() -> None:
    result = elementwise_close(
        np.array([3.0e38], dtype=np.float32),
        np.array([-3.0e38], dtype=np.float32),
        atol=0.0,
        rtol=2.1,
        equal_nan=False,
        xp=np,
    )
    assert result.tolist() == [True]


@numpy_available
def test_elementwise_close_small_float_does_not_warn() -> None:
    with np.errstate(all="raise"):
        elementwise_close(
            np.array([60000], dtype=np.float16),
            np.array([-60000], dtype=np.float16),
            atol=0.0,
            rtol=1.5,
            equal_nan=False,
            xp=np,
        )


@numpy_available
def test_elementwise_close_complex64_tolerance_in_double_precision() -> None:
    result = elementwise_close(
        np.array([1 + 1j], dtype=np.complex64),
        np.array([1.5 + 1j], dtype=np.complex64),
        atol=0.6,
        rtol=0.0,
        equal_nan=False,
        xp=np,
    )
    assert result.tolist() == [True]


@numpy_available
@torch_available
def test_elementwise_close_torch_integers_differ_by_one_above_2_53() -> None:
    assert _torch_int_close([2**62], [2**62 + 1], atol=0.5) == [False]


@numpy_available
@torch_available
def test_elementwise_close_torch_int64_difference_wraps_around() -> None:
    assert _torch_int_close([-(2**63)], [2**63 - 1], atol=1e19) == [False]


@numpy_available
@torch_available
def test_elementwise_close_torch_int64_difference_wraps_around_within_atol() -> None:
    assert _torch_int_close([-(2**63)], [2**63 - 1], atol=1e30) == [True]


@numpy_available
@torch_available
def test_elementwise_close_torch_uint8_does_not_wrap_around() -> None:
    assert _torch_int_close([255], [0], dtype=torch.uint8, atol=254.0) == [False]


@numpy_available
@torch_available
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_elementwise_close_torch_small_float_does_not_overflow(dtype: Any) -> None:
    result = elementwise_close(
        torch.tensor([60000.0], dtype=dtype),
        torch.tensor([-60000.0], dtype=dtype),
        atol=0.0,
        rtol=1.5,
        equal_nan=False,
        xp=torch,
    )
    assert result.tolist() == [False]

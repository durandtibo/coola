from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import Mock

import pytest

from coola.equality._helpers.numpy import (
    array_close,
    array_equal,
    is_numeric_array,
    time_scalar_equal,
)
from coola.equality.config import EqualityConfig
from coola.equality.tolerance import Tolerance
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#################################
#     Tests for array_equal     #
#################################


ARRAY_EQUAL = [
    pytest.param(np.array([True, False, True], dtype=np.bool_), id="boolean"),
    pytest.param(np.array([1, 0, 1], dtype=np.byte), id="signed byte"),
    pytest.param(np.array([1, 0, 1], dtype=np.ubyte), id="unsigned byte"),
    pytest.param(np.array([1, 0, 1], dtype=np.int32), id="int32"),
    pytest.param(np.array([1, 0, 1], dtype=np.int64), id="int64"),
    pytest.param(np.array([1, 0, 1], dtype=np.uint32), id="uint32"),
    pytest.param(np.array([1, 0, 1], dtype=np.uint64), id="uint64"),
    pytest.param(np.array([1, 0, 1], dtype=np.float32), id="float32"),
    pytest.param(np.array([1, 0, 1], dtype=np.float64), id="float64"),
    pytest.param(np.array([1, 0, 1], dtype=np.complex128), id="complex128"),
    pytest.param(
        np.array([np.timedelta64(1, "D"), np.timedelta64(2, "D")], dtype=np.timedelta64),
        id="timedelta64",
    ),
    pytest.param(np.array(["2005-02-25", "2007-07-13"], dtype=np.datetime64), id="datetime64"),
    pytest.param(np.array(["polar", "bear", "meow"], dtype=np.dtype("U")), id="unicode string"),
    pytest.param(
        np.array([np.void(b"abcd"), np.void(b"efg")], dtype=np.dtype("V10")), id="raw data"
    ),
]


@numpy_available
@pytest.mark.parametrize("array", ARRAY_EQUAL)
def test_array_equal_true(array: np.ndarray, config: EqualityConfig) -> None:
    assert array_equal(array, array.copy(), config=config)


@numpy_available
@pytest.mark.parametrize("array", ARRAY_EQUAL)
def test_array_equal_true_atol(array: np.ndarray, config: EqualityConfig) -> None:
    config = replace(config, tolerance=Tolerance(atol=1e-6))
    assert array_equal(array, array.copy(), config=config)


@numpy_available
@pytest.mark.parametrize("array", ARRAY_EQUAL)
def test_array_equal_true_rtol(array: np.ndarray, config: EqualityConfig) -> None:
    config = replace(config, tolerance=Tolerance(rtol=1e-3))
    assert array_equal(array, array.copy(), config=config)


@numpy_available
def test_array_equal_false_non_numeric_array1_atol(config: EqualityConfig) -> None:
    config = replace(config, tolerance=Tolerance(atol=1e-6))
    assert not array_equal(
        np.array(["polar", "bear", "meow"]), np.array([1, 0, 1], dtype=np.int64), config=config
    )


@numpy_available
def test_array_equal_false_non_numeric_array2_atol(config: EqualityConfig) -> None:
    # array2 has a non-numeric dtype: array_equal must not blindly call
    # np.allclose on it (it would raise a TypeError) and must return False instead.
    config = replace(config, tolerance=Tolerance(atol=1e-6))
    assert not array_equal(
        np.array([1, 0, 1], dtype=np.int64), np.array(["polar", "bear", "meow"]), config=config
    )


@numpy_available
def test_array_equal_false_non_numeric_array2_rtol(config: EqualityConfig) -> None:
    config = replace(config, tolerance=Tolerance(rtol=1e-3))
    assert not array_equal(
        np.array([1, 0, 1], dtype=np.int64), np.array(["polar", "bear", "meow"]), config=config
    )


######################################
#     Tests for is_numeric_array     #
######################################


@numpy_available
@pytest.mark.parametrize(
    "array",
    [
        pytest.param(np.array([True, False, True], dtype=np.bool_), id="boolean"),
        pytest.param(np.array([1, 0, 1], dtype=np.byte), id="signed byte"),
        pytest.param(np.array([1, 0, 1], dtype=np.ubyte), id="unsigned byte"),
        pytest.param(np.array([1, 0, 1], dtype=np.int32), id="int32"),
        pytest.param(np.array([1, 0, 1], dtype=np.int64), id="int64"),
        pytest.param(np.array([1, 0, 1], dtype=np.uint32), id="uint32"),
        pytest.param(np.array([1, 0, 1], dtype=np.uint64), id="uint64"),
        pytest.param(np.array([1, 0, 1], dtype=np.float32), id="float32"),
        pytest.param(np.array([1, 0, 1], dtype=np.float64), id="float64"),
        pytest.param(np.array([1, 0, 1], dtype=np.complex128), id="complex128"),
    ],
)
def test_is_numeric_array_true(array: np.ndarray) -> None:
    assert is_numeric_array(array)


@numpy_available
@pytest.mark.parametrize(
    "array",
    [
        pytest.param(
            np.array([np.timedelta64(1, "D"), np.timedelta64(2, "D")], dtype=np.timedelta64),
            id="timedelta64",
        ),
        pytest.param(np.array(["2005-02-25", "2007-07-13"], dtype=np.datetime64), id="datetime64"),
        pytest.param(np.array(["polar", "bear", "meow"], dtype=np.dtype("U")), id="unicode string"),
        pytest.param(
            np.array([np.void(b"abcd"), np.void(b"efg")], dtype=np.dtype("V10")), id="raw data"
        ),
    ],
)
def test_is_numeric_array_false(array: np.ndarray) -> None:
    assert not is_numeric_array(array)


@numpy_available
@pytest.mark.parametrize("dtype", ["U3", "S3", "O"])
def test_array_equal_equal_nan_non_numeric_dtype(dtype: str, config: EqualityConfig) -> None:
    config = replace(config, equal_nan=True)
    assert array_equal(np.array(["a", "b"], dtype=dtype), np.array(["a", "b"], dtype=dtype), config)


@numpy_available
@pytest.mark.parametrize("dtype", ["U3", "S3", "O"])
def test_array_equal_equal_nan_non_numeric_dtype_different(
    dtype: str, config: EqualityConfig
) -> None:
    config = replace(config, equal_nan=True)
    assert not array_equal(
        np.array(["a", "b"], dtype=dtype), np.array(["a", "c"], dtype=dtype), config
    )


@numpy_available
def test_array_equal_equal_nan_datetime_dtype(config: EqualityConfig) -> None:
    config = replace(config, equal_nan=True)
    assert array_equal(
        np.array(["2020-01-01", "NaT"], dtype="datetime64[D]"),
        np.array(["2020-01-01", "NaT"], dtype="datetime64[D]"),
        config,
    )


@numpy_available
def test_array_equal_rtol_scales_with_the_second_array() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert array_equal(np.array([100.0]), np.array([101.005]), config)


@numpy_available
def test_array_equal_rtol_scales_with_the_first_array() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert array_equal(np.array([101.005]), np.array([100.0]), config)


@numpy_available
def test_array_equal_rtol_outside_the_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not array_equal(np.array([101.5]), np.array([100.0]), config)


@numpy_available
def test_array_equal_rtol_outside_the_tolerance_reversed() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not array_equal(np.array([100.0]), np.array([101.5]), config)


@numpy_available
def test_array_equal_rtol_each_element_in_its_own_direction() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert array_equal(np.array([100.0, 101.005]), np.array([101.005, 100.0]), config)


@numpy_available
def test_array_equal_rtol_each_element_in_its_own_direction_reversed() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert array_equal(np.array([101.005, 100.0]), np.array([100.0, 101.005]), config)


@numpy_available
def test_array_equal_rtol_one_element_outside_the_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not array_equal(np.array([100.0, 101.005]), np.array([101.005, 105.0]), config)


@numpy_available
def test_array_equal_unsigned_integers_do_not_wrap_around() -> None:
    config = EqualityConfig(tolerance=Tolerance(atol=1.0))
    assert not array_equal(np.array([0], dtype=np.uint8), np.array([200], dtype=np.uint8), config)


@numpy_available
def test_array_equal_unsigned_integers_within_atol() -> None:
    config = EqualityConfig(tolerance=Tolerance(atol=1.0))
    assert array_equal(np.array([5], dtype=np.uint8), np.array([4], dtype=np.uint8), config)


@numpy_available
def test_array_equal_integers_within_rtol_reversed() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.02))
    assert array_equal(np.array([101, 100]), np.array([100, 101]), config)


@numpy_available
def test_array_equal_equal_infinities_with_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(atol=1.0))
    assert array_equal(np.array([float("inf")]), np.array([float("inf")]), config)


@numpy_available
def test_array_equal_different_infinities_with_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(atol=1e300))
    assert not array_equal(np.array([float("inf")]), np.array([float("-inf")]), config)


@numpy_available
def test_array_equal_complex_with_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(atol=0.6))
    assert array_equal(np.array([1 + 1j]), np.array([1.5 + 1j]), config)


@numpy_available
def test_array_equal_different_shape_with_tolerance() -> None:
    assert not array_equal(np.ones(2), np.ones(3), EqualityConfig(tolerance=Tolerance(atol=1.0)))


###################################
#     Tests for time_scalar_equal #
###################################


@numpy_available
def test_time_scalar_equal_timedelta_same_value_different_unit(config: EqualityConfig) -> None:
    assert time_scalar_equal(np.timedelta64(1, "s"), np.timedelta64(1000, "ms"), config)


@numpy_available
def test_time_scalar_equal_timedelta_different(config: EqualityConfig) -> None:
    assert not time_scalar_equal(np.timedelta64(1, "s"), np.timedelta64(2, "s"), config)


@numpy_available
def test_time_scalar_equal_timedelta_atol_in_seconds() -> None:
    assert time_scalar_equal(
        np.timedelta64(1, "s"),
        np.timedelta64(1500, "ms"),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@numpy_available
def test_time_scalar_equal_timedelta_outside_atol() -> None:
    assert not time_scalar_equal(
        np.timedelta64(1, "s"),
        np.timedelta64(3, "s"),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@numpy_available
def test_time_scalar_equal_timedelta_rtol() -> None:
    assert time_scalar_equal(
        np.timedelta64(100, "s"),
        np.timedelta64(101, "s"),
        EqualityConfig(tolerance=Tolerance(rtol=0.02)),
    )


@numpy_available
def test_time_scalar_equal_datetime_same(config: EqualityConfig) -> None:
    assert time_scalar_equal(np.datetime64("2020-01-01"), np.datetime64("2020-01-01"), config)


@numpy_available
def test_time_scalar_equal_datetime_has_no_tolerance() -> None:
    assert not time_scalar_equal(
        np.datetime64("2020-01-01"),
        np.datetime64("2020-01-02"),
        EqualityConfig(tolerance=Tolerance(atol=1e9)),
    )


@numpy_available
@pytest.mark.parametrize("value", [np.timedelta64("NaT", "s"), np.datetime64("NaT", "s")])
def test_time_scalar_equal_nat_equal_nan(value: Any) -> None:
    assert time_scalar_equal(value, value.copy(), EqualityConfig(equal_nan=True))


@numpy_available
@pytest.mark.parametrize("value", [np.timedelta64("NaT", "s"), np.datetime64("NaT", "s")])
def test_time_scalar_equal_nat_not_equal_nan(value: Any, config: EqualityConfig) -> None:
    assert not time_scalar_equal(value, value.copy(), config)


@numpy_available
def test_time_scalar_equal_nat_and_value_equal_nan() -> None:
    assert not time_scalar_equal(
        np.timedelta64("NaT", "s"), np.timedelta64(1, "s"), EqualityConfig(equal_nan=True)
    )


@numpy_available
def test_time_scalar_equal_value_and_nat_equal_nan() -> None:
    assert not time_scalar_equal(
        np.timedelta64(1, "s"), np.timedelta64("NaT", "s"), EqualityConfig(equal_nan=True)
    )


@numpy_available
def test_array_equal_timedelta_atol_in_seconds() -> None:
    assert array_equal(
        np.array([1000, 2000], dtype="m8[ms]"),
        np.array([1500, 2000], dtype="m8[ms]"),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@numpy_available
def test_array_equal_timedelta_outside_atol() -> None:
    assert not array_equal(
        np.array([1000], dtype="m8[ms]"),
        np.array([3000], dtype="m8[ms]"),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@numpy_available
def test_array_equal_timedelta_rtol() -> None:
    assert array_equal(
        np.array([100], dtype="m8[s]"),
        np.array([101], dtype="m8[s]"),
        EqualityConfig(tolerance=Tolerance(rtol=0.02)),
    )


@numpy_available
def test_array_equal_timedelta_exact_without_tolerance(config: EqualityConfig) -> None:
    assert not array_equal(
        np.array([1000], dtype="m8[ms]"), np.array([1500], dtype="m8[ms]"), config
    )


@numpy_available
def test_array_equal_timedelta_different_shape() -> None:
    assert not array_equal(
        np.array([1, 1], dtype="m8[s]"),
        np.array([1], dtype="m8[s]"),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@numpy_available
def test_array_equal_timedelta_nat_equal_nan() -> None:
    array = np.array(["NaT", 1], dtype="m8[s]")
    assert array_equal(
        array, array.copy(), EqualityConfig(equal_nan=True, tolerance=Tolerance(atol=1.0))
    )


@numpy_available
def test_array_equal_timedelta_nat_not_equal_nan() -> None:
    array = np.array(["NaT", 1], dtype="m8[s]")
    assert not array_equal(array, array.copy(), EqualityConfig(tolerance=Tolerance(atol=1.0)))


@numpy_available
def test_array_equal_timedelta_nat_and_value_equal_nan() -> None:
    assert not array_equal(
        np.array(["NaT"], dtype="m8[s]"),
        np.array([0], dtype="m8[s]"),
        EqualityConfig(equal_nan=True, tolerance=Tolerance(atol=1.0)),
    )


@numpy_available
def test_array_equal_datetime_has_no_tolerance() -> None:
    assert not array_equal(
        np.array(["2020-01-01"], dtype="M8[D]"),
        np.array(["2020-01-02"], dtype="M8[D]"),
        EqualityConfig(tolerance=Tolerance(atol=1e9)),
    )


@numpy_available
def test_array_equal_bool_has_no_atol() -> None:
    assert not array_equal(
        np.array([True]), np.array([False]), EqualityConfig(tolerance=Tolerance(atol=1.0))
    )


@numpy_available
def test_array_equal_bool_has_no_rtol() -> None:
    assert not array_equal(
        np.array([True]), np.array([False]), EqualityConfig(tolerance=Tolerance(rtol=1.0))
    )


@numpy_available
def test_array_equal_bool_equal_with_tolerance() -> None:
    assert array_equal(
        np.array([True, False]),
        np.array([True, False]),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@numpy_available
def test_array_close_exact() -> None:
    assert array_close(np.array([1.0, 2.0]), np.array([1.0, 3.0]), EqualityConfig()).tolist() == [
        True,
        False,
    ]


@numpy_available
def test_array_close_nan_without_equal_nan() -> None:
    array = np.array([float("nan")])
    assert array_close(array, array.copy(), EqualityConfig()).tolist() == [False]


@numpy_available
def test_array_close_nan_with_equal_nan() -> None:
    array = np.array([float("nan")])
    assert array_close(array, array.copy(), EqualityConfig(equal_nan=True)).tolist() == [True]


@numpy_available
def test_array_close_nat_with_equal_nan() -> None:
    array = np.array(["NaT"], dtype="M8[s]")
    assert array_close(array, array.copy(), EqualityConfig(equal_nan=True)).tolist() == [True]


@numpy_available
def test_array_close_strings_with_equal_nan() -> None:
    assert array_close(
        np.array(["a", "b"]), np.array(["a", "c"]), EqualityConfig(equal_nan=True)
    ).tolist() == [True, False]


@numpy_available
def test_array_close_with_tolerance() -> None:
    assert array_close(
        np.array([1.0, 2.0]), np.array([1.05, 3.0]), EqualityConfig(tolerance=Tolerance(atol=0.1))
    ).tolist() == [True, False]


@numpy_available
def test_array_close_durations_with_tolerance() -> None:
    assert array_close(
        np.array([1, 2], dtype="m8[s]"),
        np.array([1, 9], dtype="m8[s]"),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    ).tolist() == [True, False]


@numpy_available
def test_array_close_booleans_have_no_tolerance() -> None:
    assert array_close(
        np.array([True, False]),
        np.array([False, False]),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    ).tolist() == [False, True]

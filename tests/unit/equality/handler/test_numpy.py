from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    NumpyArrayEqualHandler,
    NumpyMaskedArrayDataHandler,
    NumpyTimeScalarEqualHandler,
    TrueHandler,
)
from coola.equality.handler.numpy import array_equal, is_numeric_array, time_scalar_equal
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available
from tests.unit.equality.tester.test_numpy import NUMPY_ARRAY_EQUAL_TOLERANCE

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if TYPE_CHECKING:
    from tests.unit.equality.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


############################################
#     Tests for NumpyArrayEqualHandler     #
############################################


def test_numpy_array_equal_handler_repr() -> None:
    assert repr(NumpyArrayEqualHandler()) == "NumpyArrayEqualHandler()"


def test_numpy_array_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(NumpyArrayEqualHandler(FalseHandler()))
        == "NumpyArrayEqualHandler(next_handler=FalseHandler())"
    )


def test_numpy_array_equal_handler_str() -> None:
    assert str(NumpyArrayEqualHandler()) == "NumpyArrayEqualHandler()"


def test_numpy_array_equal_handler_str_with_next_handler() -> None:
    assert str(NumpyArrayEqualHandler(FalseHandler())) == "NumpyArrayEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(NumpyArrayEqualHandler(), NumpyArrayEqualHandler(), id="without next handler"),
        pytest.param(
            NumpyArrayEqualHandler(FalseHandler()),
            NumpyArrayEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_numpy_array_equal_handler_equal_true(
    handler1: NumpyArrayEqualHandler, handler2: NumpyArrayEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            NumpyArrayEqualHandler(TrueHandler()),
            NumpyArrayEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            NumpyArrayEqualHandler(),
            NumpyArrayEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            NumpyArrayEqualHandler(FalseHandler()),
            NumpyArrayEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(NumpyArrayEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_numpy_array_equal_handler_equal_false(
    handler1: NumpyArrayEqualHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_numpy_array_equal_handler_equal_false_different_type_child() -> None:
    class Child(NumpyArrayEqualHandler): ...

    assert not NumpyArrayEqualHandler().equal(Child())


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (np.ones(shape=(2, 3), dtype=float), np.ones(shape=(2, 3), dtype=float)),
        (np.ones(shape=(2, 3), dtype=int), np.ones(shape=(2, 3), dtype=int)),
        (np.ones(shape=(2, 3, 4), dtype=bool), np.ones(shape=(2, 3, 4), dtype=bool)),
    ],
)
def test_numpy_array_equal_handler_handle_true(
    actual: np.ndarray, expected: np.ndarray, config: EqualityConfig
) -> None:
    assert NumpyArrayEqualHandler().handle(actual, expected, config)


@numpy_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (np.ones(shape=(2, 3)), np.ones(shape=(3, 2))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 1))),
        (np.ones(shape=(2, 3)), np.ones(shape=(2, 3, 1))),
    ],
)
def test_numpy_array_equal_handler_handle_false(
    actual: np.ndarray, expected: np.ndarray, config: EqualityConfig
) -> None:
    assert not NumpyArrayEqualHandler().handle(actual, expected, config)


@numpy_available
def test_numpy_array_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not NumpyArrayEqualHandler().handle(
        np.array([0.0, np.nan, np.nan, 1.2]), np.array([0.0, np.nan, np.nan, 1.2]), config
    )


@numpy_available
def test_numpy_array_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert NumpyArrayEqualHandler().handle(
        np.array([0.0, np.nan, np.nan, 1.2]), np.array([0.0, np.nan, np.nan, 1.2]), config
    )


@numpy_available
def test_numpy_array_equal_handler_handle_false_show_difference(config: EqualityConfig) -> None:
    config.show_difference = True
    assert not NumpyArrayEqualHandler().handle(
        actual=np.ones(shape=(2, 3)), expected=np.ones(shape=(3, 2)), config=config
    )


@numpy_available
def test_numpy_array_equal_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        NumpyArrayEqualHandler().handle(
            actual=np.ones(shape=(2, 3)), expected=np.ones(shape=(3, 2)), config=config
        )
    assert caplog.messages[0].startswith("numpy.ndarrays are different:")


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL_TOLERANCE)
def test_numpy_array_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert NumpyArrayEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_numpy_array_equal_handle_set_next_handler() -> None:
    handler = NumpyArrayEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_numpy_array_equal_handle_set_next_handler_none() -> None:
    handler = NumpyArrayEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_numpy_array_equal_handle_set_next_handler_incorrect() -> None:
    handler = NumpyArrayEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


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
    config.atol = 1e-6
    assert array_equal(array, array.copy(), config=config)


@numpy_available
@pytest.mark.parametrize("array", ARRAY_EQUAL)
def test_array_equal_true_rtol(array: np.ndarray, config: EqualityConfig) -> None:
    config.rtol = 1e-3
    assert array_equal(array, array.copy(), config=config)


@numpy_available
def test_array_equal_false_non_numeric_array1_atol(config: EqualityConfig) -> None:
    config.atol = 1e-6
    assert not array_equal(
        np.array(["polar", "bear", "meow"]), np.array([1, 0, 1], dtype=np.int64), config=config
    )


@numpy_available
def test_array_equal_false_non_numeric_array2_atol(config: EqualityConfig) -> None:
    # array2 has a non-numeric dtype: array_equal must not blindly call
    # np.allclose on it (it would raise a TypeError) and must return False instead.
    config.atol = 1e-6
    assert not array_equal(
        np.array([1, 0, 1], dtype=np.int64), np.array(["polar", "bear", "meow"]), config=config
    )


@numpy_available
def test_array_equal_false_non_numeric_array2_rtol(config: EqualityConfig) -> None:
    config.rtol = 1e-3
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
    config.equal_nan = True
    assert array_equal(np.array(["a", "b"], dtype=dtype), np.array(["a", "b"], dtype=dtype), config)


@numpy_available
@pytest.mark.parametrize("dtype", ["U3", "S3", "O"])
def test_array_equal_equal_nan_non_numeric_dtype_different(
    dtype: str, config: EqualityConfig
) -> None:
    config.equal_nan = True
    assert not array_equal(
        np.array(["a", "b"], dtype=dtype), np.array(["a", "c"], dtype=dtype), config
    )


@numpy_available
def test_array_equal_equal_nan_datetime_dtype(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert array_equal(
        np.array(["2020-01-01", "NaT"], dtype="datetime64[D]"),
        np.array(["2020-01-01", "NaT"], dtype="datetime64[D]"),
        config,
    )


@numpy_available
def test_array_equal_object_dtype_equal_nan() -> None:
    array1 = np.array([float("nan")], dtype=object)
    array2 = np.array([float("nan")], dtype=object)
    assert array_equal(array1, array2, EqualityConfig(equal_nan=True))


@numpy_available
def test_array_equal_object_dtype_not_equal_nan() -> None:
    array1 = np.array([float("nan")], dtype=object)
    array2 = np.array([float("nan")], dtype=object)
    assert not array_equal(array1, array2, EqualityConfig(equal_nan=False))


def _object_array(second: np.ndarray) -> np.ndarray:
    array = np.empty(2, dtype=object)
    array[0], array[1] = np.ones(2), second
    return array


@numpy_available
def test_array_equal_object_dtype_nested_arrays() -> None:
    assert array_equal(_object_array(np.zeros(3)), _object_array(np.zeros(3)), EqualityConfig())


@numpy_available
def test_array_equal_object_dtype_nested_arrays_different() -> None:
    assert not array_equal(_object_array(np.zeros(3)), _object_array(np.ones(3)), EqualityConfig())


@numpy_available
def test_array_equal_object_dtype_different_shape(config: EqualityConfig) -> None:
    assert not array_equal(np.array([1, 2], dtype=object), np.array([[1, 2]], dtype=object), config)


@numpy_available
def test_array_equal_rtol_scales_with_the_second_array() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert array_equal(np.array([100.0]), np.array([101.005]), config)


@numpy_available
def test_array_equal_rtol_scales_with_the_first_array() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert array_equal(np.array([101.005]), np.array([100.0]), config)


@numpy_available
def test_array_equal_rtol_outside_the_tolerance() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert not array_equal(np.array([101.5]), np.array([100.0]), config)


@numpy_available
def test_array_equal_rtol_outside_the_tolerance_reversed() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert not array_equal(np.array([100.0]), np.array([101.5]), config)


@numpy_available
def test_array_equal_rtol_each_element_in_its_own_direction() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert array_equal(np.array([100.0, 101.005]), np.array([101.005, 100.0]), config)


@numpy_available
def test_array_equal_rtol_each_element_in_its_own_direction_reversed() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert array_equal(np.array([101.005, 100.0]), np.array([100.0, 101.005]), config)


@numpy_available
def test_array_equal_rtol_one_element_outside_the_tolerance() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert not array_equal(np.array([100.0, 101.005]), np.array([101.005, 105.0]), config)


@numpy_available
def test_array_equal_unsigned_integers_do_not_wrap_around() -> None:
    config = EqualityConfig(atol=1.0)
    assert not array_equal(np.array([0], dtype=np.uint8), np.array([200], dtype=np.uint8), config)


@numpy_available
def test_array_equal_unsigned_integers_within_atol() -> None:
    config = EqualityConfig(atol=1.0)
    assert array_equal(np.array([5], dtype=np.uint8), np.array([4], dtype=np.uint8), config)


@numpy_available
def test_array_equal_integers_within_rtol_reversed() -> None:
    config = EqualityConfig(rtol=0.02)
    assert array_equal(np.array([101, 100]), np.array([100, 101]), config)


@numpy_available
def test_array_equal_equal_infinities_with_tolerance() -> None:
    config = EqualityConfig(atol=1.0)
    assert array_equal(np.array([float("inf")]), np.array([float("inf")]), config)


@numpy_available
def test_array_equal_different_infinities_with_tolerance() -> None:
    config = EqualityConfig(atol=1e300)
    assert not array_equal(np.array([float("inf")]), np.array([float("-inf")]), config)


@numpy_available
def test_array_equal_complex_with_tolerance() -> None:
    config = EqualityConfig(atol=0.6)
    assert array_equal(np.array([1 + 1j]), np.array([1.5 + 1j]), config)


@numpy_available
def test_array_equal_different_shape_with_tolerance() -> None:
    assert not array_equal(np.ones(2), np.ones(3), EqualityConfig(atol=1.0))


################################################
#     Tests for NumpyTimeScalarEqualHandler    #
################################################


@numpy_available
def test_numpy_time_scalar_equal_handler_repr() -> None:
    assert repr(NumpyTimeScalarEqualHandler()).startswith("NumpyTimeScalarEqualHandler(")


@numpy_available
def test_numpy_time_scalar_equal_handler_equal_true(config: EqualityConfig) -> None:
    assert NumpyTimeScalarEqualHandler().handle(
        np.timedelta64(1, "s"), np.timedelta64(1, "s"), config
    )


@numpy_available
def test_numpy_time_scalar_equal_handler_equal_false(config: EqualityConfig) -> None:
    assert not NumpyTimeScalarEqualHandler().handle(
        np.timedelta64(1, "s"), np.timedelta64(2, "s"), config
    )


@numpy_available
def test_numpy_time_scalar_equal_handler_equal_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        assert not NumpyTimeScalarEqualHandler().handle(
            np.timedelta64(1, "s"), np.timedelta64(2, "s"), config
        )
    assert caplog.messages[0].startswith("numpy time scalars are different")


@numpy_available
def test_numpy_time_scalar_equal_handler_equal_false_no_report(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NumpyTimeScalarEqualHandler().handle(
            np.timedelta64(1, "s"), np.timedelta64(2, "s"), config
        )
    assert not caplog.messages


@numpy_available
def test_numpy_time_scalar_equal_handler_set_next_handler() -> None:
    handler = NumpyTimeScalarEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


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
        np.timedelta64(1, "s"), np.timedelta64(1500, "ms"), EqualityConfig(atol=1.0)
    )


@numpy_available
def test_time_scalar_equal_timedelta_outside_atol() -> None:
    assert not time_scalar_equal(
        np.timedelta64(1, "s"), np.timedelta64(3, "s"), EqualityConfig(atol=1.0)
    )


@numpy_available
def test_time_scalar_equal_timedelta_rtol() -> None:
    assert time_scalar_equal(
        np.timedelta64(100, "s"), np.timedelta64(101, "s"), EqualityConfig(rtol=0.02)
    )


@numpy_available
def test_time_scalar_equal_datetime_same(config: EqualityConfig) -> None:
    assert time_scalar_equal(np.datetime64("2020-01-01"), np.datetime64("2020-01-01"), config)


@numpy_available
def test_time_scalar_equal_datetime_has_no_tolerance() -> None:
    assert not time_scalar_equal(
        np.datetime64("2020-01-01"), np.datetime64("2020-01-02"), EqualityConfig(atol=1e9)
    )


@numpy_available
@pytest.mark.parametrize("value", [np.timedelta64("NaT", "s"), np.datetime64("NaT", "s")])
def test_time_scalar_equal_nat_equal_nan(value: object) -> None:
    assert time_scalar_equal(value, value.copy(), EqualityConfig(equal_nan=True))


@numpy_available
@pytest.mark.parametrize("value", [np.timedelta64("NaT", "s"), np.datetime64("NaT", "s")])
def test_time_scalar_equal_nat_not_equal_nan(value: object, config: EqualityConfig) -> None:
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


####################################################
#     Tests for NumpyMaskedArrayDataHandler        #
####################################################


@numpy_available
def test_numpy_masked_array_data_handler_repr() -> None:
    assert repr(NumpyMaskedArrayDataHandler()).startswith("NumpyMaskedArrayDataHandler(")


@numpy_available
def test_numpy_masked_array_data_handler_equal_same_values(config: EqualityConfig) -> None:
    assert NumpyMaskedArrayDataHandler(next_handler=TrueHandler()).handle(
        np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 1]), config
    )


@numpy_available
def test_numpy_masked_array_data_handler_ignores_the_data_under_the_mask(
    config: EqualityConfig,
) -> None:
    assert NumpyMaskedArrayDataHandler(next_handler=TrueHandler()).handle(
        np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 3], mask=[0, 1]), config
    )


@numpy_available
def test_numpy_masked_array_data_handler_no_mask_equals_all_false_mask(
    config: EqualityConfig,
) -> None:
    assert NumpyMaskedArrayDataHandler(next_handler=TrueHandler()).handle(
        np.ma.array([1, 2]), np.ma.array([1, 2], mask=[0, 0]), config
    )


@numpy_available
def test_numpy_masked_array_data_handler_different_mask(config: EqualityConfig) -> None:
    assert not NumpyMaskedArrayDataHandler(next_handler=TrueHandler()).handle(
        np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[1, 0]), config
    )


@numpy_available
def test_numpy_masked_array_data_handler_different_unmasked_values(config: EqualityConfig) -> None:
    assert not NumpyMaskedArrayDataHandler(next_handler=TrueHandler()).handle(
        np.ma.array([1, 2], mask=[0, 1]), np.ma.array([3, 2], mask=[0, 1]), config
    )


@numpy_available
def test_numpy_masked_array_data_handler_different_mask_and_data_reports_the_mask(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        NumpyMaskedArrayDataHandler().handle(
            np.ma.array([1, 2], mask=[0, 1]), np.ma.array([3, 2], mask=[1, 0]), config
        )
    assert caplog.messages[-1].startswith("objects have different mask:")


@numpy_available
def test_numpy_masked_array_data_handler_different_values_reports_the_data(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        NumpyMaskedArrayDataHandler().handle(
            np.ma.array([1, 2], mask=[0, 1]), np.ma.array([3, 2], mask=[0, 1]), config
        )
    assert caplog.messages[-1].startswith("objects have different data:")


@numpy_available
def test_numpy_masked_array_data_handler_tolerance() -> None:
    assert NumpyMaskedArrayDataHandler(next_handler=TrueHandler()).handle(
        np.ma.array([1.0, 9.0], mask=[0, 1]),
        np.ma.array([1.001, 0.0], mask=[0, 1]),
        EqualityConfig(atol=0.01),
    )


@numpy_available
def test_numpy_masked_array_data_handler_equal_nan_unmasked() -> None:
    assert NumpyMaskedArrayDataHandler(next_handler=TrueHandler()).handle(
        np.ma.array([float("nan"), 1.0], mask=[0, 1]),
        np.ma.array([float("nan"), 2.0], mask=[0, 1]),
        EqualityConfig(equal_nan=True),
    )


@numpy_available
def test_numpy_masked_array_data_handler_calls_the_next_handler(config: EqualityConfig) -> None:
    assert not NumpyMaskedArrayDataHandler(next_handler=FalseHandler()).handle(
        np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 1]), config
    )


@numpy_available
def test_numpy_masked_array_data_handler_0d(config: EqualityConfig) -> None:
    assert NumpyMaskedArrayDataHandler(next_handler=TrueHandler()).handle(
        np.ma.array(5, mask=True), np.ma.array(6, mask=True), config
    )

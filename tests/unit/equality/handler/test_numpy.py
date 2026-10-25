from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, NumpyArrayEqualHandler, TrueHandler
from coola.equality.handler.numpy import array_equal, is_numeric_array
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
def test_array_equal_timedelta_atol_in_seconds() -> None:
    assert array_equal(
        np.array([1000, 2000], dtype="m8[ms]"),
        np.array([1500, 2000], dtype="m8[ms]"),
        EqualityConfig(atol=1.0),
    )


@numpy_available
def test_array_equal_timedelta_outside_atol() -> None:
    assert not array_equal(
        np.array([1000], dtype="m8[ms]"), np.array([3000], dtype="m8[ms]"), EqualityConfig(atol=1.0)
    )


@numpy_available
def test_array_equal_timedelta_rtol() -> None:
    assert array_equal(
        np.array([100], dtype="m8[s]"), np.array([101], dtype="m8[s]"), EqualityConfig(rtol=0.02)
    )


@numpy_available
def test_array_equal_timedelta_exact_without_tolerance(config: EqualityConfig) -> None:
    assert not array_equal(
        np.array([1000], dtype="m8[ms]"), np.array([1500], dtype="m8[ms]"), config
    )


@numpy_available
def test_array_equal_timedelta_different_shape() -> None:
    assert not array_equal(
        np.array([1, 1], dtype="m8[s]"), np.array([1], dtype="m8[s]"), EqualityConfig(atol=1.0)
    )


@numpy_available
def test_array_equal_timedelta_nat_equal_nan() -> None:
    array = np.array(["NaT", 1], dtype="m8[s]")
    assert array_equal(array, array.copy(), EqualityConfig(atol=1.0, equal_nan=True))


@numpy_available
def test_array_equal_timedelta_nat_not_equal_nan() -> None:
    array = np.array(["NaT", 1], dtype="m8[s]")
    assert not array_equal(array, array.copy(), EqualityConfig(atol=1.0))


@numpy_available
def test_array_equal_timedelta_nat_and_value_equal_nan() -> None:
    assert not array_equal(
        np.array(["NaT"], dtype="m8[s]"),
        np.array([0], dtype="m8[s]"),
        EqualityConfig(atol=1.0, equal_nan=True),
    )


@numpy_available
def test_array_equal_datetime_has_no_tolerance() -> None:
    assert not array_equal(
        np.array(["2020-01-01"], dtype="M8[D]"),
        np.array(["2020-01-02"], dtype="M8[D]"),
        EqualityConfig(atol=1e9),
    )

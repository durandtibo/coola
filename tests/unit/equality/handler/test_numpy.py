from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, NumpyArrayEqualHandler
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


def test_numpy_array_equal_handler_equal_true() -> None:
    assert NumpyArrayEqualHandler().equal(NumpyArrayEqualHandler())


def test_numpy_array_equal_handler_equal_false_different_type() -> None:
    assert not NumpyArrayEqualHandler().equal(FalseHandler())


def test_numpy_array_equal_handler_equal_false_different_type_child() -> None:
    class Child(NumpyArrayEqualHandler): ...

    assert not NumpyArrayEqualHandler().equal(Child())


def test_numpy_array_equal_handler_repr() -> None:
    assert repr(NumpyArrayEqualHandler()).startswith("NumpyArrayEqualHandler(")


def test_numpy_array_equal_handler_str() -> None:
    assert str(NumpyArrayEqualHandler()).startswith("NumpyArrayEqualHandler(")


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
def test_numpy_array_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = NumpyArrayEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=np.ones(shape=(2, 3)), expected=np.ones(shape=(3, 2)), config=config
        )
        assert caplog.messages[0].startswith("numpy.ndarrays have different elements:")


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


def test_numpy_array_equal_handler_set_next_handler() -> None:
    NumpyArrayEqualHandler().set_next_handler(FalseHandler())


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

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, PyarrowArrayEqualHandler
from coola.equality.handlers.pyarrow_ import array_equal
from coola.equality.testers import EqualityTester
from coola.testing import pyarrow_available
from coola.utils.imports import is_pyarrow_available
from tests.unit.equality.comparators.test_pyarrow import (
    PYARROW_ARRAY_EQUAL,
    PYARROW_ARRAY_EQUAL_TOLERANCE,
    PYARROW_ARRAY_NOT_EQUAL,
    PYARROW_ARRAY_NOT_EQUAL_TOLERANCE,
)

if TYPE_CHECKING:
    from tests.unit.equality.comparators.utils import ExamplePair

if is_pyarrow_available():
    import pyarrow as pa
else:
    pa = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##############################################
#     Tests for PyarrowArrayEqualHandler     #
##############################################


def test_pyarrow_array_equal_handler_eq_true() -> None:
    assert PyarrowArrayEqualHandler() == PyarrowArrayEqualHandler()


def test_pyarrow_array_equal_handler_eq_false() -> None:
    assert PyarrowArrayEqualHandler() != FalseHandler()


def test_pyarrow_array_equal_handler_repr() -> None:
    assert repr(PyarrowArrayEqualHandler()).startswith("PyarrowArrayEqualHandler(")


def test_pyarrow_array_equal_handler_str() -> None:
    assert str(PyarrowArrayEqualHandler()).startswith("PyarrowArrayEqualHandler(")


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_ARRAY_EQUAL)
def test_pyarrow_array_equal_handler_handle_true(
    example: ExamplePair, config: EqualityConfig
) -> None:
    assert PyarrowArrayEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_ARRAY_NOT_EQUAL)
def test_pyarrow_array_equal_handler_handle_false(
    example: ExamplePair, config: EqualityConfig
) -> None:
    assert not PyarrowArrayEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
def test_pyarrow_array_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PyarrowArrayEqualHandler().handle(
        pa.array([0.0, float("nan"), float("nan"), 1.2]),
        pa.array([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pyarrow_available
def test_pyarrow_array_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    with warnings.catch_warnings(record=True) as w:
        assert not PyarrowArrayEqualHandler().handle(
            pa.array([0.0, float("nan"), float("nan"), 1.2]),
            pa.array([0.0, float("nan"), float("nan"), 1.2]),
            config,
        )

        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "equal_nan is not supported" in str(w[-1].message)


@pyarrow_available
def test_pyarrow_array_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PyarrowArrayEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 4.0], type=pa.float64()),
            config=config,
        )
        assert caplog.messages[0].startswith("pyarrow.Arrays have different elements:")


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_ARRAY_EQUAL_TOLERANCE)
def test_pyarrow_array_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    with warnings.catch_warnings(record=True) as w:
        assert PyarrowArrayEqualHandler().handle(
            actual=example.actual, expected=example.expected, config=config
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "tol is not supported" in str(w[-1].message)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_ARRAY_NOT_EQUAL_TOLERANCE)
def test_pyarrow_array_equal_handler_handle_false_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    with warnings.catch_warnings(record=True) as w:
        assert not PyarrowArrayEqualHandler().handle(
            actual=example.actual, expected=example.expected, config=config
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "tol is not supported" in str(w[-1].message)


def test_pyarrow_array_equal_handler_set_next_handler() -> None:
    PyarrowArrayEqualHandler().set_next_handler(FalseHandler())


#################################
#     Tests for array_equal     #
#################################


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_ARRAY_EQUAL)
def test_array_equal_true(example: ExamplePair, config: EqualityConfig) -> None:
    assert array_equal(example.actual, example.expected, config=config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_ARRAY_NOT_EQUAL)
def test_array_equal_false(example: ExamplePair, config: EqualityConfig) -> None:
    assert not array_equal(example.actual, example.expected, config=config)

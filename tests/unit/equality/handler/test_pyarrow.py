from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, PyarrowEqualHandler
from coola.equality.handler.pyarrow import object_equal
from coola.testing.fixtures import pyarrow_available
from coola.utils.imports import is_pyarrow_available
from tests.unit.equality.tester.test_pyarrow import (
    PYARROW_EQUAL,
    PYARROW_EQUAL_TOLERANCE,
    PYARROW_NOT_EQUAL,
    PYARROW_NOT_EQUAL_TOLERANCE,
)

if TYPE_CHECKING:
    from tests.unit.equality.utils import ExamplePair

if is_pyarrow_available():
    import pyarrow as pa
else:
    pa = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#########################################
#     Tests for PyarrowEqualHandler     #
#########################################


def test_pyarrow_equal_handler__eq__true() -> None:
    assert PyarrowEqualHandler() == PyarrowEqualHandler()


def test_pyarrow_equal_handler__eq__false_different_type() -> None:
    assert PyarrowEqualHandler() != FalseHandler()


def test_pyarrow_equal_handler__eq__false_different_type_child() -> None:
    class Child(PyarrowEqualHandler): ...

    assert PyarrowEqualHandler() != Child()


def test_pyarrow_equal_handler_repr() -> None:
    assert repr(PyarrowEqualHandler()).startswith("PyarrowEqualHandler(")


def test_pyarrow_equal_handler_str() -> None:
    assert str(PyarrowEqualHandler()).startswith("PyarrowEqualHandler(")


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_pyarrow_equal_handler_handle_true(example: ExamplePair, config: EqualityConfig) -> None:
    assert PyarrowEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_pyarrow_equal_handler_handle_false(example: ExamplePair, config: EqualityConfig) -> None:
    assert not PyarrowEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
def test_pyarrow_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PyarrowEqualHandler().handle(
        pa.array([0.0, float("nan"), float("nan"), 1.2]),
        pa.array([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pyarrow_available
def test_pyarrow_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    with warnings.catch_warnings(record=True) as w:
        # Not equal because equal_nan is ignored
        assert not PyarrowEqualHandler().handle(
            pa.array([0.0, float("nan"), float("nan"), 1.2]),
            pa.array([0.0, float("nan"), float("nan"), 1.2]),
            config,
        )

        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "equal_nan is ignored because it is not supported" in str(w[-1].message)


@pyarrow_available
def test_pyarrow_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PyarrowEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 4.0], type=pa.float64()),
            config=config,
        )
        assert caplog.messages[0].startswith("objects are different")


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL_TOLERANCE)
def test_pyarrow_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    with warnings.catch_warnings(record=True) as w:
        assert PyarrowEqualHandler().handle(
            actual=example.actual, expected=example.expected, config=config
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "tol is ignored because it is not supported" in str(w[-1].message)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL_TOLERANCE)
def test_pyarrow_equal_handler_handle_false_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    with warnings.catch_warnings(record=True) as w:
        assert not PyarrowEqualHandler().handle(
            actual=example.actual, expected=example.expected, config=config
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "tol is ignored because it is not supported" in str(w[-1].message)


def test_pyarrow_equal_handler_set_next_handler() -> None:
    PyarrowEqualHandler().set_next_handler(FalseHandler())


##################################
#     Tests for object_equal     #
##################################


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_object_equal_true(example: ExamplePair, config: EqualityConfig) -> None:
    assert object_equal(example.actual, example.expected, config=config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_object_equal_false(example: ExamplePair, config: EqualityConfig) -> None:
    assert not object_equal(example.actual, example.expected, config=config)

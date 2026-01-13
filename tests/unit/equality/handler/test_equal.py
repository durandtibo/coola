from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import EqualNanHandler, FalseHandler, TrueHandler
from coola.equality.handler.equal import EqualHandler

if TYPE_CHECKING:
    from coola.equality.handler.equal import SupportsEqual, SupportsEqualNan


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


class MyFloat:
    def __init__(self, value: float) -> None:
        self._value = value

    def equal(self, other: float) -> bool:
        return self._value == other


class MyFloatNan:
    def __init__(self, value: float) -> None:
        self._value = value

    def equal(self, other: float, equal_nan: bool = False) -> bool:
        if equal_nan and math.isnan(self._value) and math.isnan(other):
            return True
        return self._value == other


##################################
#     Tests for EqualHandler     #
##################################


def test_equal_handler__eq__true() -> None:
    assert EqualHandler() == EqualHandler()


def test_equal_handler__eq__false_different_type() -> None:
    assert EqualHandler() != FalseHandler()


def test_equal_handler__eq__false_different_type_child() -> None:
    class Child(EqualHandler): ...

    assert EqualHandler() != Child()


def test_equal_handler_repr() -> None:
    assert repr(EqualHandler()).startswith("EqualHandler(")


def test_equal_handler_str() -> None:
    assert str(EqualHandler()).startswith("EqualHandler(")


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(MyFloat(42), 42), (MyFloat(0), 0)],
)
def test_equal_handler_handle_true(
    actual: SupportsEqual, expected: Any, config: EqualityConfig
) -> None:
    assert EqualHandler().handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (MyFloat(42), 1),
        (MyFloat(0), float("nan")),
        (MyFloat(float("nan")), float("nan")),
    ],
)
def test_equal_handler_handle_false(
    actual: SupportsEqual, expected: Any, config: EqualityConfig
) -> None:
    assert not EqualHandler().handle(actual, expected, config)


def test_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = EqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual=MyFloat(42), expected=1, config=config)
        assert caplog.messages[-1].startswith("objects are not equal:")


@pytest.mark.parametrize("equal_nan", [True, False])
def test_equal_handler_handle_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert not EqualHandler().handle(MyFloat(float("nan")), float("nan"), config)


def test_equal_handler_set_next_handler() -> None:
    EqualHandler().set_next_handler(TrueHandler())


#####################################
#     Tests for EqualNanHandler     #
#####################################


def test_equal_nan_handler__eq__true() -> None:
    assert EqualNanHandler() == EqualNanHandler()


def test_equal_nan_handler__eq__false_different_type() -> None:
    assert EqualNanHandler() != FalseHandler()


def test_equal_nan_handler__eq__false_different_type_child() -> None:
    class Child(EqualNanHandler): ...

    assert EqualNanHandler() != Child()


def test_equal_nan_handler_repr() -> None:
    assert repr(EqualNanHandler()).startswith("EqualNanHandler(")


def test_equal_nan_handler_str() -> None:
    assert str(EqualNanHandler()).startswith("EqualNanHandler(")


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(MyFloatNan(42), 42), (MyFloatNan(0), 0)],
)
def test_equal_nan_handler_handle_true(
    actual: SupportsEqualNan, expected: Any, config: EqualityConfig
) -> None:
    assert EqualNanHandler().handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (MyFloatNan(42), 1),
        (MyFloatNan(0), float("nan")),
        (MyFloatNan(float("nan")), float("nan")),
    ],
)
def test_equal_nan_handler_handle_false(
    actual: SupportsEqualNan, expected: Any, config: EqualityConfig
) -> None:
    assert not EqualNanHandler().handle(actual, expected, config)


def test_equal_nan_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = EqualNanHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual=MyFloatNan(42), expected=1, config=config)
        assert caplog.messages[-1].startswith("objects are not equal:")


@pytest.mark.parametrize("equal_nan", [True, False])
def test_equal_nan_handler_handle_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert EqualNanHandler().handle(MyFloatNan(float("nan")), float("nan"), config) == equal_nan


def test_equal_nan_handler_set_next_handler() -> None:
    EqualNanHandler().set_next_handler(TrueHandler())

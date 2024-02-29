from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import EqualHandler, FalseHandler, TrueHandler
from coola.equality.testers import EqualityTester

if TYPE_CHECKING:
    from coola.equality.handlers.equal import SupportsEqual


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


class MyFloat:
    def __init__(self, value: float) -> None:
        self._value = value

    def equal(self, other: float, equal_nan: bool = False) -> bool:
        if equal_nan and math.isnan(self._value) and math.isnan(other):
            return True
        return self._value == other


#####################################
#     Tests for EqualHandler     #
#####################################


def test_same_data_handler_eq_true() -> None:
    assert EqualHandler() == EqualHandler()


def test_same_data_handler_eq_false() -> None:
    assert EqualHandler() != FalseHandler()


def test_same_data_handler_repr() -> None:
    assert repr(EqualHandler()).startswith("EqualHandler(")


def test_same_data_handler_str() -> None:
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
    assert EqualHandler().handle(MyFloat(float("nan")), float("nan"), config) == equal_nan


def test_equal_handler_set_next_handler() -> None:
    EqualHandler().set_next_handler(TrueHandler())

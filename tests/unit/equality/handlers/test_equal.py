from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import EqualNanHandler, FalseHandler, TrueHandler
from coola.equality.testers import EqualityTester

if TYPE_CHECKING:
    from coola.equality.handlers.equal import SupportsEqualNan


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


class MyFloatNan:
    def __init__(self, value: float) -> None:
        self._value = value

    def equal(self, other: float, equal_nan: bool = False) -> bool:
        if equal_nan and math.isnan(self._value) and math.isnan(other):
            return True
        return self._value == other


#####################################
#     Tests for EqualNanHandler     #
#####################################


def test_same_data_handler_eq_true() -> None:
    assert EqualNanHandler() == EqualNanHandler()


def test_same_data_handler_eq_false() -> None:
    assert EqualNanHandler() != FalseHandler()


def test_same_data_handler_repr() -> None:
    assert repr(EqualNanHandler()).startswith("EqualNanHandler(")


def test_same_data_handler_str() -> None:
    assert str(EqualNanHandler()).startswith("EqualNanHandler(")


@pytest.mark.parametrize(
    ("actual", "expected"),
    [(MyFloatNan(42), 42), (MyFloatNan(0), 0)],
)
def test_equal_handler_handle_true(
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
def test_equal_handler_handle_false(
    actual: SupportsEqualNan, expected: Any, config: EqualityConfig
) -> None:
    assert not EqualNanHandler().handle(actual, expected, config)


def test_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = EqualNanHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(actual=MyFloatNan(42), expected=1, config=config)
        assert caplog.messages[-1].startswith("objects are not equal:")


@pytest.mark.parametrize("equal_nan", [True, False])
def test_equal_handler_handle_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert EqualNanHandler().handle(MyFloatNan(float("nan")), float("nan"), config) == equal_nan


def test_equal_handler_set_next_handler() -> None:
    EqualNanHandler().set_next_handler(TrueHandler())

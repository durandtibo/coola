from __future__ import annotations

import logging
from typing import Any

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, FloatEqualHandler
from coola.equality.testers import EqualityTester


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#######################################
#     Tests for FloatEqualHandler     #
#######################################


def test_float_equal_handler_eq_true() -> None:
    assert FloatEqualHandler() == FloatEqualHandler()


def test_float_equal_handler_eq_false() -> None:
    assert FloatEqualHandler() != FalseHandler()


def test_float_equal_handler_repr() -> None:
    assert repr(FloatEqualHandler()).startswith("FloatEqualHandler(")


def test_float_equal_handler_str() -> None:
    assert str(FloatEqualHandler()).startswith("FloatEqualHandler(")


@pytest.mark.parametrize(("object1", "object2"), [(0, 0), (4.2, 4.2), (-1.0, -1.0)])
def test_float_equal_handler_handle_true(
    object1: float, object2: float, config: EqualityConfig
) -> None:
    assert FloatEqualHandler().handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"), [(0, 1), (4, 4.2), (float("nan"), 1.0), (float("nan"), float("nan"))]
)
def test_float_equal_handler_handle_false(
    object1: float, object2: Any, config: EqualityConfig
) -> None:
    assert not FloatEqualHandler().handle(object1, object2, config)


def test_float_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = FloatEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1=1.0, object2=2.0, config=config)
        assert caplog.messages[0].startswith("numbers are not equal:")


@pytest.mark.parametrize("equal_nan", [True, False])
def test_float_equal_handler_handle_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert FloatEqualHandler().handle(float("nan"), float("nan"), config) == equal_nan


def test_float_equal_handler_set_next_handler() -> None:
    FloatEqualHandler().set_next_handler(FalseHandler())

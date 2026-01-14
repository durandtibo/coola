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


def test_equal_handler_repr() -> None:
    assert repr(EqualHandler()) == "EqualHandler()"


def test_equal_handler_str() -> None:
    assert str(EqualHandler()) == "EqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(EqualHandler(), EqualHandler(), id="without next handler"),
        pytest.param(
            EqualHandler(FalseHandler()),
            EqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_equal_handler_equal_true(handler1: EqualHandler, handler2: EqualHandler) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            EqualHandler(TrueHandler()),
            EqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            EqualHandler(),
            EqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            EqualHandler(FalseHandler()),
            EqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(EqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_equal_handler_equal_false(handler1: EqualHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_equal_handler_equal_false_different_type_child() -> None:
    class Child(EqualHandler): ...

    assert not EqualHandler().equal(Child())


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
    handler = EqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_equal_handler_set_next_handler_none() -> None:
    handler = EqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_equal_handler_set_next_handler_incorrect() -> None:
    handler = EqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


#####################################
#     Tests for EqualNanHandler     #
#####################################


def test_equal_nan_handler_repr() -> None:
    assert repr(EqualNanHandler()) == "EqualNanHandler()"


def test_equal_nan_handler_str() -> None:
    assert str(EqualNanHandler()) == "EqualNanHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(EqualNanHandler(), EqualNanHandler(), id="without next handler"),
        pytest.param(
            EqualNanHandler(FalseHandler()),
            EqualNanHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_equal_nan_handler_equal_true(handler1: EqualNanHandler, handler2: EqualNanHandler) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            EqualNanHandler(TrueHandler()),
            EqualNanHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            EqualNanHandler(),
            EqualNanHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            EqualNanHandler(FalseHandler()),
            EqualNanHandler(),
            id="other next handler is none",
        ),
        pytest.param(EqualNanHandler(), FalseHandler(), id="different type"),
    ],
)
def test_equal_nan_handler_equal_false(handler1: EqualNanHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_equal_nan_handler_equal_false_different_type_child() -> None:
    class Child(EqualNanHandler): ...

    assert not EqualNanHandler().equal(Child())


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
    handler = EqualNanHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_equal_nan_handler_set_next_handler_none() -> None:
    handler = EqualNanHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_equal_nan_handler_set_next_handler_incorrect() -> None:
    handler = EqualNanHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)

from __future__ import annotations

import logging
import math
from typing import Any

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, TrueHandler
from coola.equality.handler.tolerant import (
    SupportsTolerantEqual,
    TolerantEqualHandler,
)


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


class MyFloat:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def allclose(
        self,
        other: object,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        if type(other) is not type(self):
            return False
        if equal_nan and math.isnan(self._value) and math.isnan(other._value):
            return True
        return math.isclose(self._value, other._value, rel_tol=rtol, abs_tol=atol)

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        if type(other) is not type(self):
            return False
        if equal_nan and math.isnan(self._value) and math.isnan(other._value):
            return True
        return self._value == other._value


class NoMethods:
    """Object that implements neither allclose nor equal."""


class NoAllcloseMethod:
    """Object that implements equal but not allclose."""

    def equal(
        self,
        other: object,  # noqa: ARG002
        equal_nan: bool = False,  # noqa: ARG002
    ) -> bool:
        return False


class NoEqualMethod:
    """Object that implements allclose but not equal."""

    def allclose(
        self,
        other: object,  # noqa: ARG002
        rtol: float = 1e-5,  # noqa: ARG002
        atol: float = 1e-8,  # noqa: ARG002
        equal_nan: bool = False,  # noqa: ARG002
    ) -> bool:
        return False


class AllcloseNotCallable:
    """Object where allclose exists but is not callable."""

    allclose = "not_a_method"

    def equal(
        self,
        other: object,  # noqa: ARG002
        equal_nan: bool = False,  # noqa: ARG002
    ) -> bool:
        return False


class EqualNotCallable:
    """Object where equal exists but is not callable."""

    def allclose(
        self,
        other: object,  # noqa: ARG002
        rtol: float = 1e-5,  # noqa: ARG002
        atol: float = 1e-8,  # noqa: ARG002
        equal_nan: bool = False,  # noqa: ARG002
    ) -> bool:
        return False

    equal = "not_a_method"


############################################
#     Tests for TolerantEqualHandler       #
############################################


def test_tolerant_equal_handler_repr() -> None:
    assert repr(TolerantEqualHandler()) == "TolerantEqualHandler()"


def test_tolerant_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(TolerantEqualHandler(FalseHandler()))
        == "TolerantEqualHandler(next_handler=FalseHandler())"
    )


def test_tolerant_equal_handler_str() -> None:
    assert str(TolerantEqualHandler()) == "TolerantEqualHandler()"


def test_tolerant_equal_handler_str_with_next_handler() -> None:
    assert str(TolerantEqualHandler(FalseHandler())) == "TolerantEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(TolerantEqualHandler(), TolerantEqualHandler(), id="without next handler"),
        pytest.param(
            TolerantEqualHandler(FalseHandler()),
            TolerantEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_tolerant_equal_handler_equal_true(
    handler1: TolerantEqualHandler, handler2: TolerantEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            TolerantEqualHandler(TrueHandler()),
            TolerantEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            TolerantEqualHandler(),
            TolerantEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            TolerantEqualHandler(FalseHandler()),
            TolerantEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(TolerantEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_tolerant_equal_handler_equal_false(
    handler1: TolerantEqualHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_tolerant_equal_handler_equal_false_different_type_child() -> None:
    class Child(TolerantEqualHandler): ...

    assert not TolerantEqualHandler().equal(Child())


# --- handle: zero tolerance dispatches to equal ---


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(MyFloat(42), MyFloat(42), id="equal integers"),
        pytest.param(MyFloat(0), MyFloat(0), id="equal zeros"),
        pytest.param(MyFloat(3.14), MyFloat(3.14), id="equal floats"),
    ],
)
def test_tolerant_equal_handler_handle_true_uses_equal(
    actual: SupportsTolerantEqual, expected: Any, config: EqualityConfig
) -> None:
    assert config.atol == 0
    assert config.rtol == 0
    assert TolerantEqualHandler().handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(MyFloat(42), MyFloat(1), id="different values"),
        pytest.param(MyFloat(42), 42, id="different types"),
        pytest.param(MyFloat(0), MyFloat(float("nan")), id="number vs nan"),
        pytest.param(MyFloat(float("nan")), MyFloat(float("nan")), id="nan vs nan equal_nan=False"),
        pytest.param(NoMethods(), NoMethods(), id="neither method"),
        pytest.param(NoAllcloseMethod(), NoAllcloseMethod(), id="missing allclose"),
        pytest.param(NoEqualMethod(), NoEqualMethod(), id="missing equal"),
        pytest.param(AllcloseNotCallable(), AllcloseNotCallable(), id="allclose not callable"),
        pytest.param(EqualNotCallable(), EqualNotCallable(), id="equal not callable"),
    ],
)
def test_tolerant_equal_handler_handle_false_uses_equal(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    assert config.atol == 0
    assert config.rtol == 0
    assert not TolerantEqualHandler().handle(actual, expected, config)


# --- handle: non-zero tolerance dispatches to allclose ---


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(MyFloat(42), MyFloat(42), id="equal integers"),
        pytest.param(MyFloat(1.0), MyFloat(1.4), id="within atol"),
    ],
)
def test_tolerant_equal_handler_handle_true_uses_allclose(
    actual: SupportsTolerantEqual, expected: Any, config: EqualityConfig
) -> None:
    config.atol = 0.5
    assert TolerantEqualHandler().handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(MyFloat(42), MyFloat(1), id="different values"),
        pytest.param(MyFloat(42), 42, id="different types"),
        pytest.param(MyFloat(float("nan")), MyFloat(float("nan")), id="nan vs nan equal_nan=False"),
    ],
)
def test_tolerant_equal_handler_handle_false_uses_allclose(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    config.atol = 0.5
    assert not TolerantEqualHandler().handle(actual, expected, config)


# --- handle: NaN behaviour ---


@pytest.mark.parametrize("equal_nan", [True, False])
def test_tolerant_equal_handler_handle_equal_nan_uses_equal(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert config.atol == 0
    assert config.rtol == 0
    assert (
        TolerantEqualHandler().handle(MyFloat(float("nan")), MyFloat(float("nan")), config)
        == equal_nan
    )


@pytest.mark.parametrize("equal_nan", [True, False])
def test_tolerant_equal_handler_handle_equal_nan_uses_allclose(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.atol = 0.5
    config.equal_nan = equal_nan
    assert (
        TolerantEqualHandler().handle(MyFloat(float("nan")), MyFloat(float("nan")), config)
        == equal_nan
    )


# --- handle: tolerance dispatch boundary ---


@pytest.mark.parametrize(
    ("atol", "rtol"),
    [
        pytest.param(0.5, 0.0, id="non-zero atol"),
        pytest.param(0.0, 0.5, id="non-zero rtol"),
        pytest.param(0.5, 0.5, id="both non-zero"),
    ],
)
def test_tolerant_equal_handler_handle_dispatches_to_allclose_when_tolerances_nonzero(
    config: EqualityConfig, atol: float, rtol: float
) -> None:
    config.atol = atol
    config.rtol = rtol
    # 1.0 vs 1.4 would fail equal() but pass allclose() with atol=0.5
    assert TolerantEqualHandler().handle(MyFloat(1.0), MyFloat(1.4), config)


def test_tolerant_equal_handler_handle_dispatches_to_equal_when_tolerances_zero(
    config: EqualityConfig,
) -> None:
    config.atol = 0.0
    config.rtol = 0.0
    # 1.0 vs 1.4 passes allclose() with default tolerances but fails equal()
    assert not TolerantEqualHandler().handle(MyFloat(1.0), MyFloat(1.4), config)


# --- handle: logging ---


def test_tolerant_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        assert not TolerantEqualHandler().handle(
            actual=MyFloat(42), expected=MyFloat(1), config=config
        )
        assert caplog.messages[-1].startswith("objects are different:")


def test_tolerant_equal_handler_handle_no_log_when_equal(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        assert TolerantEqualHandler().handle(
            actual=MyFloat(42), expected=MyFloat(42), config=config
        )
    assert not caplog.messages


def test_tolerant_equal_handler_handle_no_log_when_show_difference_false(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = False
    with caplog.at_level(logging.INFO):
        assert not TolerantEqualHandler().handle(
            actual=MyFloat(42), expected=MyFloat(1), config=config
        )
    assert not caplog.messages


# --- set_next_handler ---


def test_tolerant_equal_handler_set_next_handler() -> None:
    handler = TolerantEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_tolerant_equal_handler_set_next_handler_none() -> None:
    handler = TolerantEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_tolerant_equal_handler_set_next_handler_incorrect() -> None:
    handler = TolerantEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)

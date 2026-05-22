from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    AllCloseNanHandler,
    FalseHandler,
    TrueHandler,
)

if TYPE_CHECKING:
    from coola.equality.handler.allclose import SupportsAllCloseNan

####################
#     Fixtures     #
####################


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(rtol=1e-5, atol=1e-8)


####################
#     Helpers      #
####################


class MyFloatAllCloseNan:
    """Minimal SupportsAllCloseNan implementation for testing."""

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


class NoAllcloseMethod:
    """Object that does not implement allclose."""


class AllcloseNotCallable:
    """Object where allclose exists but is not callable."""

    allclose = "not_a_method"


#########################################
#     Tests for AllCloseNanHandler      #
#########################################


def test_allclose_nan_handler_repr() -> None:
    assert repr(AllCloseNanHandler()) == "AllCloseNanHandler()"


def test_allclose_nan_handler_repr_with_next_handler() -> None:
    assert (
        repr(AllCloseNanHandler(FalseHandler()))
        == "AllCloseNanHandler(next_handler=FalseHandler())"
    )


def test_allclose_nan_handler_str() -> None:
    assert str(AllCloseNanHandler()) == "AllCloseNanHandler()"


def test_allclose_nan_handler_str_with_next_handler() -> None:
    assert str(AllCloseNanHandler(FalseHandler())) == "AllCloseNanHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(AllCloseNanHandler(), AllCloseNanHandler(), id="without next handler"),
        pytest.param(
            AllCloseNanHandler(FalseHandler()),
            AllCloseNanHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_allclose_nan_handler_equal_true(
    handler1: AllCloseNanHandler, handler2: AllCloseNanHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            AllCloseNanHandler(TrueHandler()),
            AllCloseNanHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            AllCloseNanHandler(),
            AllCloseNanHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            AllCloseNanHandler(FalseHandler()),
            AllCloseNanHandler(),
            id="other next handler is none",
        ),
        pytest.param(AllCloseNanHandler(), FalseHandler(), id="different type"),
    ],
)
def test_allclose_nan_handler_equal_false(handler1: AllCloseNanHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_allclose_nan_handler_equal_false_different_type_child() -> None:
    class Child(AllCloseNanHandler): ...

    assert not AllCloseNanHandler().equal(Child())


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(MyFloatAllCloseNan(42), MyFloatAllCloseNan(42), id="equal integers"),
        pytest.param(MyFloatAllCloseNan(0), MyFloatAllCloseNan(0), id="equal zeros"),
        pytest.param(MyFloatAllCloseNan(3.14), MyFloatAllCloseNan(3.14), id="equal floats"),
        pytest.param(
            MyFloatAllCloseNan(1.0), MyFloatAllCloseNan(1.0 + 1e-9), id="within default tolerance"
        ),
    ],
)
def test_allclose_nan_handler_handle_true(
    actual: SupportsAllCloseNan, expected: Any, config: EqualityConfig
) -> None:
    assert AllCloseNanHandler().handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(MyFloatAllCloseNan(42), MyFloatAllCloseNan(1), id="different values"),
        pytest.param(MyFloatAllCloseNan(42), 42, id="different types"),
        pytest.param(MyFloatAllCloseNan(0), MyFloatAllCloseNan(float("nan")), id="number vs nan"),
        pytest.param(
            MyFloatAllCloseNan(float("nan")),
            MyFloatAllCloseNan(float("nan")),
            id="nan vs nan equal_nan=False",
        ),
        pytest.param(NoAllcloseMethod(), NoAllcloseMethod(), id="missing allclose method"),
        pytest.param(AllcloseNotCallable(), AllcloseNotCallable(), id="allclose not callable"),
    ],
)
def test_allclose_nan_handler_handle_false(
    actual: Any, expected: Any, config: EqualityConfig
) -> None:
    assert not AllCloseNanHandler().handle(actual, expected, config)


def test_allclose_nan_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = AllCloseNanHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=MyFloatAllCloseNan(42), expected=MyFloatAllCloseNan(1), config=config
        )
        assert caplog.messages[-1].startswith("objects are different:")


def test_allclose_nan_handler_handle_no_log_when_equal(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        assert AllCloseNanHandler().handle(
            actual=MyFloatAllCloseNan(42), expected=MyFloatAllCloseNan(42), config=config
        )
    assert not caplog.messages


def test_allclose_nan_handler_handle_no_log_when_show_difference_false(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = False
    with caplog.at_level(logging.INFO):
        assert not AllCloseNanHandler().handle(
            actual=MyFloatAllCloseNan(42), expected=MyFloatAllCloseNan(1), config=config
        )
    assert not caplog.messages


@pytest.mark.parametrize("equal_nan", [True, False])
def test_allclose_nan_handler_handle_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        AllCloseNanHandler().handle(
            MyFloatAllCloseNan(float("nan")), MyFloatAllCloseNan(float("nan")), config
        )
        == equal_nan
    )


@pytest.mark.parametrize(
    ("rtol", "atol", "actual", "expected", "outcome"),
    [
        pytest.param(
            0.5, 0.0, MyFloatAllCloseNan(1.0), MyFloatAllCloseNan(1.4), True, id="within rtol"
        ),
        pytest.param(
            0.0, 0.5, MyFloatAllCloseNan(1.0), MyFloatAllCloseNan(1.4), True, id="within atol"
        ),
        pytest.param(
            0.0,
            0.0,
            MyFloatAllCloseNan(1.0),
            MyFloatAllCloseNan(1.4),
            False,
            id="outside rtol and atol",
        ),
    ],
)
def test_allclose_nan_handler_handle_tolerance(
    config: EqualityConfig,
    rtol: float,
    atol: float,
    actual: SupportsAllCloseNan,
    expected: Any,
    outcome: bool,
) -> None:
    config.rtol = rtol
    config.atol = atol
    assert AllCloseNanHandler().handle(actual, expected, config) == outcome


def test_allclose_nan_handler_set_next_handler() -> None:
    handler = AllCloseNanHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_allclose_nan_handler_set_next_handler_none() -> None:
    handler = AllCloseNanHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_allclose_nan_handler_set_next_handler_incorrect() -> None:
    handler = AllCloseNanHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)

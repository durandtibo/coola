from __future__ import annotations

import logging
from typing import Any

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, FloatEqualHandler, ScalarEqualHandler
from coola.equality.testers import EqualityTester
from tests.unit.equality.comparators.utils import ExamplePair


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


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (0, 0),
        (4.2, 4.2),
        (-1.0, -1.0),
        (float("inf"), float("inf")),
        (float("-inf"), float("-inf")),
    ],
)
def test_float_equal_handler_handle_true(
    object1: float, object2: float, config: EqualityConfig
) -> None:
    assert FloatEqualHandler().handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [(0, 1), (4, 4.2), (float("inf"), 1.0), (float("nan"), 1.0), (float("nan"), float("nan"))],
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


########################################
#     Tests for ScalarEqualHandler     #
########################################


def test_scalar_equal_handler_eq_true() -> None:
    assert ScalarEqualHandler() == ScalarEqualHandler()


def test_scalar_equal_handler_eq_false() -> None:
    assert ScalarEqualHandler() != FalseHandler()


def test_scalar_equal_handler_repr() -> None:
    assert repr(ScalarEqualHandler()).startswith("ScalarEqualHandler(")


def test_scalar_equal_handler_str() -> None:
    assert str(ScalarEqualHandler()).startswith("ScalarEqualHandler(")


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(ExamplePair(object1=4, object2=4), id="int"),
        pytest.param(ExamplePair(object1=4.2, object2=4.2), id="float"),
        pytest.param(ExamplePair(object1=-1.0, object2=-1.0), id="negative"),
        pytest.param(ExamplePair(object1=float("inf"), object2=float("inf")), id="infinity"),
        pytest.param(ExamplePair(object1=float("-inf"), object2=float("-inf")), id="-infinity"),
    ],
)
def test_scalar_equal_handler_handle_true(example: ExamplePair, config: EqualityConfig) -> None:
    assert ScalarEqualHandler().handle(example.object1, example.object2, config)


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(ExamplePair(object1=0, object2=1), id="different values - int"),
        pytest.param(ExamplePair(object1=4.0, object2=4.2), id="different values - float"),
        pytest.param(ExamplePair(object1=float("inf"), object2=4.2), id="different values - inf"),
        pytest.param(
            ExamplePair(object1=float("inf"), object2=float("-inf")), id="opposite infinity"
        ),
        pytest.param(ExamplePair(object1=float("nan"), object2=1.0), id="one nan"),
        pytest.param(ExamplePair(object1=float("nan"), object2=float("nan")), id="two nans"),
    ],
)
def test_scalar_equal_handler_handle_false(example: ExamplePair, config: EqualityConfig) -> None:
    assert not ScalarEqualHandler().handle(example.object1, example.object2, config)


def test_scalar_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = ScalarEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1=1.0, object2=2.0, config=config)
        assert caplog.messages[0].startswith("numbers are not equal:")


@pytest.mark.parametrize("equal_nan", [True, False])
def test_scalar_equal_handler_handle_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert ScalarEqualHandler().handle(float("nan"), float("nan"), config) == equal_nan


@pytest.mark.parametrize(
    ("object1", "object2", "atol"),
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 5, 10),
        (1.0, 1.0 + 1e-4, 1e-3),
        (1.0, 1.0 - 1e-4, 1e-3),
        (False, True, 1),
    ],
)
def test_scalar_equal_handler_handle_true_atol(
    object1: float,
    object2: float,
    atol: float,
    config: EqualityConfig,
) -> None:
    config.atol = atol
    config.rtol = 0.0
    assert ScalarEqualHandler().handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2", "rtol"),
    [
        (0, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (1, 5, 10),
        (1.0, 1.0 + 1e-4, 1e-3),
        (1.0, 1.0 - 1e-4, 1e-3),
        (False, True, 1),
    ],
)
def test_scalar_equal_handler_handle_true_rtol(
    object1: float,
    object2: float,
    rtol: float,
    config: EqualityConfig,
) -> None:
    config.atol = 0.0
    config.rtol = rtol
    assert ScalarEqualHandler().handle(object1, object2, config)


def test_scalar_equal_handler_set_next_handler() -> None:
    ScalarEqualHandler().set_next_handler(FalseHandler())

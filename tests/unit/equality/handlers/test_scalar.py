from __future__ import annotations

import logging

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, NanEqualHandler, ScalarEqualHandler
from coola.equality.testers import EqualityTester
from tests.unit.equality.comparators.test_scalar import SCALAR_EQUAL_TOLERANCE
from tests.unit.equality.comparators.utils import ExamplePair


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#####################################
#     Tests for NanEqualHandler     #
#####################################


def test_nan_equal_handler_eq_true() -> None:
    assert NanEqualHandler() == NanEqualHandler()


def test_nan_equal_handler_eq_false() -> None:
    assert NanEqualHandler() != FalseHandler()


def test_nan_equal_handler_repr() -> None:
    assert repr(NanEqualHandler()).startswith("NanEqualHandler(")


def test_nan_equal_handler_str() -> None:
    assert str(NanEqualHandler()) == "NanEqualHandler()"


def test_nan_equal_handler_handle_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert NanEqualHandler().handle(float("nan"), float("nan"), config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (float("nan"), float("nan")),
        (4.2, 4.2),
        (1, 0),
    ],
)
def test_nan_equal_handler_handle_false(
    object1: float, object2: float, config: EqualityConfig
) -> None:
    assert not NanEqualHandler(next_handler=FalseHandler()).handle(object1, object2, config)


def test_nan_equal_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = NanEqualHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1=42, object2=42, config=config)


def test_nan_equal_handler_set_next_handler() -> None:
    handler = NanEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_nan_equal_handler_set_next_handler_incorrect() -> None:
    handler = NanEqualHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


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
    assert not ScalarEqualHandler().handle(float("nan"), float("nan"), config)


@pytest.mark.parametrize("example", SCALAR_EQUAL_TOLERANCE)
def test_scalar_equal_handler_handle_true_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert ScalarEqualHandler().handle(example.object1, example.object2, config)


def test_scalar_equal_handler_set_next_handler() -> None:
    ScalarEqualHandler().set_next_handler(FalseHandler())

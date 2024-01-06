from __future__ import annotations

from typing import Any

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, SameObjectHandler, TrueHandler


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##################################
#     Tests for FalseHandler     #
##################################


def test_false_handler_eq_true() -> None:
    assert FalseHandler() == FalseHandler()


def test_false_handler_eq_false() -> None:
    assert FalseHandler() != TrueHandler()


def test_false_handler_str() -> None:
    assert str(FalseHandler()).startswith("FalseHandler(")


@pytest.mark.parametrize(
    ("object1", "object2"), [(0, 0), (4.2, 4.2), ("abc", "abc"), (0, 1), (4, 4.0), ("abc", "ABC")]
)
def test_false_handler_handle(object1: Any, object2: Any, config: EqualityConfig) -> None:
    assert not FalseHandler().handle(object1, object2, config)


def test_false_handler_set_next_handler() -> None:
    FalseHandler().set_next_handler(TrueHandler())


##################################
#     Tests for TrueHandler     #
##################################


def test_true_handler_eq_true() -> None:
    assert TrueHandler() == TrueHandler()


def test_true_handler_eq_false() -> None:
    assert TrueHandler() != FalseHandler()


def test_true_handler_str() -> None:
    assert str(TrueHandler()).startswith("TrueHandler(")


@pytest.mark.parametrize(
    ("object1", "object2"), [(0, 0), (4.2, 4.2), ("abc", "abc"), (0, 1), (4, 4.0), ("abc", "ABC")]
)
def test_true_handler_handle(object1: Any, object2: Any, config: EqualityConfig) -> None:
    assert TrueHandler().handle(object1, object2, config)


def test_true_handler_set_next_handler() -> None:
    TrueHandler().set_next_handler(FalseHandler())


#######################################
#     Tests for SameObjectHandler     #
#######################################


def test_same_object_handler_eq_true() -> None:
    assert SameObjectHandler() == SameObjectHandler()


def test_same_object_handler_eq_false() -> None:
    assert SameObjectHandler() != FalseHandler()


def test_same_object_handler_str() -> None:
    assert str(SameObjectHandler()).startswith("SameObjectHandler(")


@pytest.mark.parametrize(("object1", "object2"), [(0, 0), (4.2, 4.2), ("abc", "abc")])
def test_same_object_handler_handle_true(
    object1: Any, object2: Any, config: EqualityConfig
) -> None:
    assert SameObjectHandler(next_handler=FalseHandler()).handle(object1, object2, config)


@pytest.mark.parametrize(("object1", "object2"), [(0, 1), (4, 4.0), ("abc", "ABC")])
def test_same_object_handler_handle_false(
    object1: Any, object2: Any, config: EqualityConfig
) -> None:
    assert not SameObjectHandler(next_handler=FalseHandler()).handle(object1, object2, config)


def test_same_object_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    with pytest.raises(RuntimeError, match="The next handler is not defined"):
        handler.handle(object1="abc", object2="ABC", config=config)


def test_same_object_handler_set_next_handler() -> None:
    handler = SameObjectHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_object_handler_set_next_handler_incorrect() -> None:
    handler = SameObjectHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)

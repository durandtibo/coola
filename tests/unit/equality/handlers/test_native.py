from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import (
    FalseHandler,
    ObjectEqualHandler,
    SameLengthHandler,
    SameObjectHandler,
    SameTypeHandler,
    TrueHandler,
)

if TYPE_CHECKING:
    from collections.abc import Sized


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


########################################
#     Tests for ObjectEqualHandler     #
########################################


def test_object_equal_handler_eq_true() -> None:
    assert ObjectEqualHandler() == ObjectEqualHandler()


def test_object_equal_handler_eq_false() -> None:
    assert ObjectEqualHandler() != FalseHandler()


def test_object_equal_handler_str() -> None:
    assert str(ObjectEqualHandler()).startswith("ObjectEqualHandler(")


@pytest.mark.parametrize(("object1", "object2"), [(0, 0), (4.2, 4.2), ("abc", "abc")])
def test_object_equal_handler_handle_true(
    object1: Any, object2: Any, config: EqualityConfig
) -> None:
    assert ObjectEqualHandler().handle(object1, object2, config)


@pytest.mark.parametrize(("object1", "object2"), [(0, 1), (4, 4.2), ("abc", "ABC")])
def test_object_equal_handler_handle_false(
    object1: Any, object2: Any, config: EqualityConfig
) -> None:
    assert not ObjectEqualHandler().handle(object1, object2, config)


def test_object_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = ObjectEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1=[1, 2, 3], object2=[1, 2, 3, 4], config=config)
        assert caplog.messages[0].startswith("objects are different:")


def test_object_equal_handler_set_next_handler() -> None:
    ObjectEqualHandler().set_next_handler(FalseHandler())


#######################################
#     Tests for SameLengthHandler     #
#######################################


def test_same_length_handler_eq_true() -> None:
    assert SameLengthHandler() == SameLengthHandler()


def test_same_length_handler_eq_false() -> None:
    assert SameLengthHandler() != FalseHandler()


def test_same_length_handler_str() -> None:
    assert str(SameLengthHandler()).startswith("SameLengthHandler(")


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([], []),
        ([1, 2, 3], [4, 5, 6]),
        ((1, 2, 3), (4, 5, 6)),
        ({1, 2, 3}, {4, 5, 6}),
        ("abc", "abc"),
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_same_length_handler_handle_true(
    object1: Sized, object2: Sized, config: EqualityConfig
) -> None:
    assert SameLengthHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        ([1, 2], [1, 2, 3]),
        ((4, 5), (4, 5, None)),
        ({"a", "b", "c"}, {"a"}),
    ],
)
def test_same_length_handler_handle_false(
    object1: Sized, object2: Sized, config: EqualityConfig
) -> None:
    assert not SameLengthHandler().handle(object1, object2, config)


def test_same_length_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameLengthHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1=[1, 2, 3], object2=[1, 2, 3, 4], config=config)
        assert caplog.messages[0].startswith("objects have different lengths:")


def test_same_length_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameLengthHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1=[1, 2, 3], object2=[1, 2, 3], config=config)


def test_same_length_handler_set_next_handler() -> None:
    handler = SameLengthHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_length_handler_set_next_handler_incorrect() -> None:
    handler = SameLengthHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


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
    assert SameObjectHandler().handle(object1, object2, config)


@pytest.mark.parametrize(("object1", "object2"), [(0, 1), (4, 4.0), ("abc", "ABC")])
def test_same_object_handler_handle_false(
    object1: Any, object2: Any, config: EqualityConfig
) -> None:
    assert not SameObjectHandler(next_handler=FalseHandler()).handle(object1, object2, config)


def test_same_object_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameObjectHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1="abc", object2="ABC", config=config)


def test_same_object_handler_set_next_handler() -> None:
    handler = SameObjectHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_object_handler_set_next_handler_incorrect() -> None:
    handler = SameObjectHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)


#####################################
#     Tests for SameTypeHandler     #
#####################################


def test_same_type_handler_eq_true() -> None:
    assert SameTypeHandler() == SameTypeHandler()


def test_same_type_handler_eq_false() -> None:
    assert SameTypeHandler() != FalseHandler()


def test_same_type_handler_str() -> None:
    assert str(SameTypeHandler()).startswith("SameTypeHandler(")


@pytest.mark.parametrize(("object1", "object2"), [(0, 0), (4.2, 4.2), ("abc", "abc")])
def test_same_type_handler_handle_true(object1: Any, object2: Any, config: EqualityConfig) -> None:
    assert SameTypeHandler(next_handler=TrueHandler()).handle(object1, object2, config)


@pytest.mark.parametrize(("object1", "object2"), [(0, "abc"), (4, 4.0), (None, 0)])
def test_same_type_handler_handle_false(object1: Any, object2: Any, config: EqualityConfig) -> None:
    assert not SameTypeHandler().handle(object1, object2, config)


def test_same_type_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = SameTypeHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1=0, object2="abc", config=config)
        assert caplog.messages[0].startswith("objects have different types:")


def test_same_type_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = SameTypeHandler()
    with pytest.raises(RuntimeError, match="next handler is not defined"):
        handler.handle(object1="abc", object2="ABC", config=config)


def test_same_type_handler_set_next_handler() -> None:
    handler = SameTypeHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler == FalseHandler()


def test_same_type_handler_set_next_handler_incorrect() -> None:
    handler = SameTypeHandler()
    with pytest.raises(TypeError, match="Incorrect type for `handler`."):
        handler.set_next_handler(None)

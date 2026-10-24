from __future__ import annotations

import functools
import logging
from typing import Any

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    ObjectSameAttributesHandler,
    TrueHandler,
    get_object_attributes,
)
from coola.equality.tester import (
    DefaultEqualityTester,
    EqualityTesterRegistry,
    ObjectAttributesEqualityTester,
    ScalarEqualityTester,
)


class Point:
    def __init__(self, x: Any, y: Any = 0) -> None:
        self.x = x
        self.y = y


class Slotted:
    __slots__ = ("x", "y")

    def __init__(self, x: Any, y: Any = 0) -> None:
        self.x = x
        self.y = y


class SlottedChild(Slotted):
    __slots__ = ("z",)

    def __init__(self, x: Any, y: Any = 0, z: Any = 0) -> None:
        super().__init__(x, y)
        self.z = z


class SlotsAsString:
    __slots__ = "x"

    def __init__(self, x: Any) -> None:
        self.x = x


class SlottedWithDict:
    __slots__ = ("__dict__", "x")

    def __init__(self, x: Any, y: Any) -> None:
        self.x = x
        self.y = y


class Node:
    def __init__(self) -> None:
        self.child: Any = None


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


###########################################
#     Tests for get_object_attributes     #
###########################################


def test_get_object_attributes_plain_object() -> None:
    assert get_object_attributes(Point(1, 2)) == {"x": 1, "y": 2}


def test_get_object_attributes_object_without_attribute() -> None:
    assert get_object_attributes(object()) == {}


def test_get_object_attributes_slots() -> None:
    assert get_object_attributes(Slotted(1, 2)) == {"x": 1, "y": 2}


def test_get_object_attributes_slot_that_is_not_set() -> None:
    obj = Slotted(1)
    del obj.y
    assert get_object_attributes(obj) == {"x": 1}


def test_get_object_attributes_slots_of_the_parent_classes() -> None:
    assert get_object_attributes(SlottedChild(1, 2, 3)) == {"x": 1, "y": 2, "z": 3}


def test_get_object_attributes_slots_as_a_string() -> None:
    assert get_object_attributes(SlotsAsString(1)) == {"x": 1}


def test_get_object_attributes_slots_and_dict() -> None:
    assert get_object_attributes(SlottedWithDict(1, 2)) == {"x": 1, "y": 2}


def test_get_object_attributes_exception() -> None:
    assert get_object_attributes(ValueError("a", 1)) == {"args": ("a", 1)}


def test_get_object_attributes_exception_with_attribute() -> None:
    error = KeyError("a")
    error.code = 3  # ty: ignore[unresolved-attribute]
    assert get_object_attributes(error) == {"args": ("a",), "code": 3}


def test_get_object_attributes_partial() -> None:
    assert get_object_attributes(functools.partial(print, 1, sep="-")) == {
        "func": print,
        "args": (1,),
        "keywords": {"sep": "-"},
    }


def test_get_object_attributes_does_not_return_the_dict_of_the_object() -> None:
    obj = Point(1)
    attributes = get_object_attributes(obj)
    attributes["z"] = 3
    assert not hasattr(obj, "z")


###############################################
#     Tests for ObjectSameAttributesHandler  #
###############################################


def test_object_same_attributes_handler_equal_true() -> None:
    assert ObjectSameAttributesHandler().equal(ObjectSameAttributesHandler())


def test_object_same_attributes_handler_equal_false_different_type() -> None:
    assert not ObjectSameAttributesHandler().equal(TrueHandler())


def test_object_same_attributes_handler_equal_false_different_next() -> None:
    assert not ObjectSameAttributesHandler(TrueHandler()).equal(ObjectSameAttributesHandler())


def test_object_same_attributes_handler_str() -> None:
    assert str(ObjectSameAttributesHandler()) == "ObjectSameAttributesHandler()"


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (Point(1, 2), Point(1, 2)),
        (Point([1, 2]), Point([1, 2])),
        (Slotted(1, 2), Slotted(1, 2)),
        (SlottedChild(1, 2, 3), SlottedChild(1, 2, 3)),
        (ValueError("a"), ValueError("a")),
        (functools.partial(print, 1), functools.partial(print, 1)),
        (object(), object()),
    ],
)
def test_object_same_attributes_handler_handle_true(
    config: EqualityConfig, actual: Any, expected: Any
) -> None:
    assert ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (Point(1, 2), Point(1, 3)),
        (Point([1, 2]), Point([1, 3])),
        (Slotted(1, 2), Slotted(1, 3)),
        (SlottedChild(1, 2, 3), SlottedChild(1, 2, 4)),
        (ValueError("a"), ValueError("b")),
        (functools.partial(print, 1), functools.partial(print, 2)),
        (functools.partial(print, 1), functools.partial(len, 1)),
    ],
)
def test_object_same_attributes_handler_handle_false(
    config: EqualityConfig, actual: Any, expected: Any
) -> None:
    assert not ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(
        actual, expected, config
    )


def test_object_same_attributes_handler_handle_different_attribute_names(
    config: EqualityConfig,
) -> None:
    other = Point(1, 2)
    other.z = 3  # ty: ignore[unresolved-attribute]
    assert not ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(
        Point(1, 2), other, config
    )


def test_object_same_attributes_handler_handle_calls_next(config: EqualityConfig) -> None:
    assert not ObjectSameAttributesHandler(next_handler=FalseHandler()).handle(
        Point(1), Point(1), config
    )


def test_object_same_attributes_handler_handle_nan_different_by_default(
    config: EqualityConfig,
) -> None:
    assert not ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(
        Point(float("nan")), Point(float("nan")), config
    )


def test_object_same_attributes_handler_handle_equal_nan() -> None:
    config = EqualityConfig(equal_nan=True)
    assert ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(
        Point(float("nan")), Point(float("nan")), config
    )


def test_object_same_attributes_handler_handle_tolerance() -> None:
    config = EqualityConfig(atol=1e-6)
    assert ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(
        Point(1.0), Point(1.0 + 1e-9), config
    )


def test_object_same_attributes_handler_handle_cycle() -> None:
    actual, expected = Node(), Node()
    actual.child = actual
    expected.child = expected
    registry = EqualityTesterRegistry(
        {object: DefaultEqualityTester(), Node: ObjectAttributesEqualityTester()}
    )
    config = EqualityConfig(registry=registry)
    assert ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(actual, expected, config)


def test_object_same_attributes_handler_handle_show_difference_logs_the_attribute(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = EqualityConfig(show_difference=True)
    with caplog.at_level(logging.INFO):
        ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(
            Point(1, 2), Point(1, 3), config
        )
    assert caplog.messages[-1] == "objects have different y: 2 vs 3"


def test_object_same_attributes_handler_handle_show_difference_reports_the_attribute_path() -> None:
    reports: list[tuple[str, tuple]] = []
    config = EqualityConfig(
        show_difference=True, reporter=lambda msg, path: reports.append((msg, path))
    )
    ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(Point(1, 2), Point(1, 3), config)
    assert reports[-1] == ("objects have different y: 2 vs 3", ("y",))


def test_object_same_attributes_handler_handle_show_difference_different_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    other = Point(1, 2)
    other.z = 3  # ty: ignore[unresolved-attribute]
    config = EqualityConfig(show_difference=True)
    with caplog.at_level(logging.INFO):
        ObjectSameAttributesHandler(next_handler=TrueHandler()).handle(Point(1, 2), other, config)
    assert caplog.messages == ["objects have different attributes: ['x', 'y'] vs ['x', 'y', 'z']"]


#####################################################
#     Tests for ObjectAttributesEqualityTester     #
#####################################################


@pytest.fixture
def registry() -> EqualityTesterRegistry:
    return EqualityTesterRegistry(
        {
            object: DefaultEqualityTester(),
            float: ScalarEqualityTester(),
            Point: ObjectAttributesEqualityTester(),
            Slotted: ObjectAttributesEqualityTester(),
            BaseException: ObjectAttributesEqualityTester(),
            functools.partial: ObjectAttributesEqualityTester(),
        }
    )


def test_object_attributes_equality_tester_repr() -> None:
    assert repr(ObjectAttributesEqualityTester()) == "ObjectAttributesEqualityTester()"


def test_object_attributes_equality_tester_equal_true() -> None:
    assert ObjectAttributesEqualityTester().equal(ObjectAttributesEqualityTester())


def test_object_attributes_equality_tester_equal_false_different_type() -> None:
    assert not ObjectAttributesEqualityTester().equal(DefaultEqualityTester())


def test_object_attributes_equality_tester_same_object(config: EqualityConfig) -> None:
    obj = Point(1)
    assert ObjectAttributesEqualityTester().objects_are_equal(obj, obj, config)


def test_object_attributes_equality_tester_equal_attributes(config: EqualityConfig) -> None:
    assert ObjectAttributesEqualityTester().objects_are_equal(Point(1), Point(1), config)


def test_object_attributes_equality_tester_different_attributes(config: EqualityConfig) -> None:
    assert not ObjectAttributesEqualityTester().objects_are_equal(Point(1), Point(2), config)


def test_object_attributes_equality_tester_different_types(config: EqualityConfig) -> None:
    assert not ObjectAttributesEqualityTester().objects_are_equal(Point(1), Slotted(1), config)


def test_object_attributes_equality_tester_different_types_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = EqualityConfig(show_difference=True)
    with caplog.at_level(logging.INFO):
        ObjectAttributesEqualityTester().objects_are_equal(Point(1), Slotted(1), config)
    assert "different types" in caplog.text

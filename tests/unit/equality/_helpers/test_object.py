from __future__ import annotations

import functools
from typing import Any

import pytest

from coola.equality._helpers import (
    get_object_attributes,
)
from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    DefaultEqualityTester,
    EqualityTesterRegistry,
    ObjectAttributesEqualityTester,
    ScalarEqualityTester,
)
from tests.unit.equality.utils import ExplainingConfig, explaining, messages, run_tester


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


def _init_slots_as_string(self: Any, x: Any) -> None:
    self.x = x


# ``__slots__`` is built from a dict because a plain ``__slots__ = "x"`` is rejected by ruff.
SlotsAsString = type("SlotsAsString", (), {"__slots__": "x", "__init__": _init_slots_as_string})


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
    assert ObjectAttributesEqualityTester() == ObjectAttributesEqualityTester()


def test_object_attributes_equality_tester_equal_false_different_type() -> None:
    assert ObjectAttributesEqualityTester() != DefaultEqualityTester()


def test_object_attributes_equality_tester_same_object(config: EqualityConfig) -> None:
    obj = Point(1)
    assert run_tester(ObjectAttributesEqualityTester(), obj, obj, config)


def test_object_attributes_equality_tester_equal_attributes(config: EqualityConfig) -> None:
    assert run_tester(ObjectAttributesEqualityTester(), Point(1), Point(1), config)


def test_object_attributes_equality_tester_different_attributes(config: EqualityConfig) -> None:
    assert not run_tester(ObjectAttributesEqualityTester(), Point(1), Point(2), config)


def test_object_attributes_equality_tester_different_types(config: EqualityConfig) -> None:
    assert not run_tester(ObjectAttributesEqualityTester(), Point(1), Slotted(1), config)


def test_object_attributes_equality_tester_different_types_logs() -> None:
    config = ExplainingConfig()
    run_tester(ObjectAttributesEqualityTester(), Point(1), Slotted(1), config)
    assert "different types" in "\n".join(messages(config))


def test_object_attributes_equality_tester_different_attribute_names(
    config: EqualityConfig,
) -> None:
    other = Point(1, 2)
    other.z = 3  # ty: ignore[unresolved-attribute]
    assert not run_tester(ObjectAttributesEqualityTester(), Point(1, 2), other, config)


def test_object_attributes_equality_tester_different_attribute_names_reason(
    config: EqualityConfig,
) -> None:
    other = Point(1, 2)
    other.z = 3  # ty: ignore[unresolved-attribute]
    config = explaining(config)
    run_tester(ObjectAttributesEqualityTester(), Point(1, 2), other, config)
    assert messages(config) == ["objects have different attributes: ['x', 'y'] vs ['x', 'y', 'z']"]

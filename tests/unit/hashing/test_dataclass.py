from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from coola.hashing import DataclassHasher, HasherRegistry, Node, get_default_registry
from tests.unit.hashing.helpers import FixedHasher, hash_with


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Other:
    x: int
    y: int


@dataclass
class Line:
    start: Point
    end: Point


@pytest.fixture
def registry() -> HasherRegistry:
    registry = HasherRegistry()
    registry.register(dict, get_default_registry().find_hasher(dict))
    registry.register(int, get_default_registry().find_hasher(int))
    registry.register(str, get_default_registry().find_hasher(str))
    registry.register(Point, DataclassHasher())
    return registry


#####################################
#     Tests for DataclassHasher     #
#####################################


def test_dataclass_hasher_repr() -> None:
    assert repr(DataclassHasher()) == "DataclassHasher()"


def test_dataclass_hasher_str() -> None:
    assert str(DataclassHasher()) == "DataclassHasher()"


def test_dataclass_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = DataclassHasher()
    assert hash_with(hasher, Point(1, 2), registry) == hash_with(hasher, Point(1, 2), registry)


def test_dataclass_hasher_hash_different_values_differ(registry: HasherRegistry) -> None:
    hasher = DataclassHasher()
    assert hash_with(hasher, Point(1, 2), registry) != hash_with(hasher, Point(1, 3), registry)


def test_dataclass_hasher_hash_depends_on_class(registry: HasherRegistry) -> None:
    hasher = DataclassHasher()
    assert hash_with(hasher, Point(1, 2), registry) != hash_with(hasher, Other(1, 2), registry)


def test_dataclass_hasher_hash_differs_from_plain_dict(registry: HasherRegistry) -> None:
    hasher = DataclassHasher()
    assert hash_with(hasher, Point(1, 2), registry) != registry.hash({"x": 1, "y": 2})


def test_dataclass_hasher_hash_uses_class_tag(registry: HasherRegistry) -> None:
    node = Node(f"dataclass:{Point.__module__}.{Point.__qualname__}", ["x", 1, "y", 2], False, 2)
    assert hash_with(DataclassHasher(), Point(1, 2), registry) == hash_with(
        FixedHasher(node), Point(1, 2), registry
    )


@pytest.mark.parametrize("length", [2, 16, 64, 128])
def test_dataclass_hasher_hash_length(length: int, registry: HasherRegistry) -> None:
    assert len(hash_with(DataclassHasher(), Point(1, 2), registry, length=length)) == length


def test_dataclass_hasher_hash_nested_dataclasses() -> None:
    default = get_default_registry()
    registry = HasherRegistry()
    registry.register_many(
        {
            dict: default.find_hasher(dict),
            int: default.find_hasher(int),
            str: default.find_hasher(str),
            Point: DataclassHasher(),
            Line: DataclassHasher(),
        }
    )
    a = Line(Point(0, 0), Point(1, 1))
    b = Line(Point(0, 0), Point(1, 2))
    assert registry.hash(a) != registry.hash(b)


def test_dataclass_hasher_hash_unregistered_field_raises(registry: HasherRegistry) -> None:
    @dataclass
    class Holder:
        value: object

    with pytest.raises(KeyError):
        hash_with(DataclassHasher(), Holder(object()), registry)


def test_dataclass_hasher_hash_not_a_dataclass_raises(registry: HasherRegistry) -> None:
    with pytest.raises(TypeError, match="Expected a dataclass instance"):
        hash_with(DataclassHasher(), 1, registry)


def test_dataclass_hasher_hash_dataclass_type_raises(registry: HasherRegistry) -> None:
    with pytest.raises(TypeError, match="Expected a dataclass instance"):
        hash_with(DataclassHasher(), Point, registry)


@dataclass
class WithCache:
    value: int
    cache: int = field(default=0, compare=False)


def test_dataclass_hasher_hash_ignores_fields_with_compare_false() -> None:
    registry = get_default_registry()
    hasher = DataclassHasher()
    assert hash_with(hasher, WithCache(1, cache=0), registry) == hash_with(
        hasher, WithCache(1, cache=5), registry
    )


def test_dataclass_hasher_hash_uses_fields_with_compare_true() -> None:
    registry = get_default_registry()
    hasher = DataclassHasher()
    assert hash_with(hasher, WithCache(1), registry) != hash_with(hasher, WithCache(2), registry)


def test_dataclass_hasher_hash_ignores_unhashable_field_with_compare_false() -> None:
    @dataclass
    class Holder:
        value: int
        other: object = field(default=None, compare=False)

    registry = get_default_registry()
    assert hash_with(DataclassHasher(), Holder(1, object()), registry) == hash_with(
        DataclassHasher(), Holder(1), registry
    )

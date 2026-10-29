from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from coola.hashing import (
    ContentHasher,
    DataclassHasher,
    HasherRegistry,
    Node,
    SupportsCoolaHash,
    get_default_registry,
)
from tests.unit.hashing.helpers import FixedHasher, encode_node, hash_with


@pytest.fixture
def registry() -> HasherRegistry:
    return get_default_registry()


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

    with pytest.raises(KeyError, match="is not registered"):
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


class ContentPoint:
    def __init__(self, x: object, y: object) -> None:
        self.x = x
        self.y = y

    def __coola_hash__(self) -> object:
        return (self.x, self.y)


class ContentOther(ContentPoint):
    pass


class NoContent:
    pass


###########################################
#     Tests for SupportsCoolaHash     #
###########################################


def test_supports_coola_hash_isinstance_true() -> None:
    assert isinstance(ContentPoint(1, 2), SupportsCoolaHash)


def test_supports_coola_hash_isinstance_false() -> None:
    assert not isinstance(NoContent(), SupportsCoolaHash)


####################################
#     Tests for ContentHasher     #
####################################


def test_content_hasher_repr() -> None:
    assert repr(ContentHasher()) == "ContentHasher()"


def test_content_hasher_str() -> None:
    assert str(ContentHasher()) == "ContentHasher()"


def test_content_hasher_encode_returns_a_node() -> None:
    assert isinstance(ContentHasher().encode(ContentPoint(1, 2)), Node)


def test_content_hasher_encode_tag_is_the_qualified_class_name() -> None:
    node = encode_node(ContentHasher(), ContentPoint(1, 2))
    assert node.tag == f"custom:{ContentPoint.__module__}.ContentPoint"


def test_content_hasher_encode_child_is_the_content() -> None:
    node = encode_node(ContentHasher(), ContentPoint(1, 2))
    assert list(node.children) == [(1, 2)]


def test_content_hasher_encode_node_is_ordered() -> None:
    assert encode_node(ContentHasher(), ContentPoint(1, 2)).ordered


def test_content_hasher_encode_calls_the_method_once() -> None:
    calls = []

    class Counting:
        def __coola_hash__(self) -> object:
            calls.append(1)
            return 1

    ContentHasher().encode(Counting())
    assert len(calls) == 1


def test_content_hasher_encode_method_error_propagates() -> None:
    class Failing:
        def __coola_hash__(self) -> object:
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        ContentHasher().encode(Failing())


def test_content_hasher_hash_equal_content(registry: HasherRegistry) -> None:
    hasher = ContentHasher()
    assert hash_with(hasher, ContentPoint(1, 2), registry) == hash_with(
        hasher, ContentPoint(1, 2), registry
    )


def test_content_hasher_hash_different_content(registry: HasherRegistry) -> None:
    hasher = ContentHasher()
    assert hash_with(hasher, ContentPoint(1, 2), registry) != hash_with(
        hasher, ContentPoint(2, 1), registry
    )


def test_content_hasher_hash_differs_from_the_content(registry: HasherRegistry) -> None:
    assert hash_with(ContentHasher(), ContentPoint(1, 2), registry) != registry.hash((1, 2))


def test_content_hasher_hash_different_classes_with_the_same_content_differ(
    registry: HasherRegistry,
) -> None:
    hasher = ContentHasher()
    assert hash_with(hasher, ContentPoint(1, 2), registry) != hash_with(
        hasher, ContentOther(1, 2), registry
    )


@pytest.mark.parametrize("length", [2, 16, 64, 128])
def test_content_hasher_hash_length(length: int, registry: HasherRegistry) -> None:
    assert len(hash_with(ContentHasher(), ContentPoint(1, 2), registry, length=length)) == length


def test_content_hasher_hash_content_with_a_missing_hasher_raises() -> None:
    with pytest.raises(KeyError, match="is not registered"):
        hash_with(ContentHasher(), ContentPoint(1, 2), HasherRegistry())


def test_content_hasher_hash_content_with_a_missing_hasher_ignore_unhashable() -> None:
    with pytest.warns(UserWarning, match="No hasher registered"):
        result = hash_with(
            ContentHasher(), ContentPoint(1, 2), HasherRegistry(), ignore_unhashable=True
        )
    assert len(result) == 64

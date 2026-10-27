from __future__ import annotations

import pytest

from coola.hashing import (
    ContentHasher,
    HasherRegistry,
    Node,
    SupportsCoolaHash,
    get_default_registry,
)
from tests.unit.hashing.helpers import hash_with


class Point:
    def __init__(self, x: object, y: object) -> None:
        self.x = x
        self.y = y

    def __coola_hash__(self) -> object:
        return (self.x, self.y)


class Other(Point):
    pass


class NoContent:
    pass


@pytest.fixture
def registry() -> HasherRegistry:
    return get_default_registry()


###########################################
#     Tests for SupportsCoolaHash     #
###########################################


def test_supports_coola_hash_isinstance_true() -> None:
    assert isinstance(Point(1, 2), SupportsCoolaHash)


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
    assert isinstance(ContentHasher().encode(Point(1, 2)), Node)


def test_content_hasher_encode_tag_is_the_qualified_class_name() -> None:
    node = ContentHasher().encode(Point(1, 2))
    assert node.tag == f"custom:{Point.__module__}.Point"


def test_content_hasher_encode_child_is_the_content() -> None:
    node = ContentHasher().encode(Point(1, 2))
    assert list(node.children) == [(1, 2)]


def test_content_hasher_encode_node_is_ordered() -> None:
    assert ContentHasher().encode(Point(1, 2)).ordered


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
    assert hash_with(hasher, Point(1, 2), registry) == hash_with(hasher, Point(1, 2), registry)


def test_content_hasher_hash_different_content(registry: HasherRegistry) -> None:
    hasher = ContentHasher()
    assert hash_with(hasher, Point(1, 2), registry) != hash_with(hasher, Point(2, 1), registry)


def test_content_hasher_hash_differs_from_the_content(registry: HasherRegistry) -> None:
    assert hash_with(ContentHasher(), Point(1, 2), registry) != registry.hash((1, 2))


def test_content_hasher_hash_different_classes_with_the_same_content_differ(
    registry: HasherRegistry,
) -> None:
    hasher = ContentHasher()
    assert hash_with(hasher, Point(1, 2), registry) != hash_with(hasher, Other(1, 2), registry)


@pytest.mark.parametrize("length", [2, 16, 64, 128])
def test_content_hasher_hash_length(length: int, registry: HasherRegistry) -> None:
    assert len(hash_with(ContentHasher(), Point(1, 2), registry, length=length)) == length


def test_content_hasher_hash_content_with_a_missing_hasher_raises() -> None:
    with pytest.raises(KeyError):
        hash_with(ContentHasher(), Point(1, 2), HasherRegistry())


def test_content_hasher_hash_content_with_a_missing_hasher_ignore_unhashable() -> None:
    with pytest.warns(UserWarning, match="No hasher registered"):
        result = hash_with(ContentHasher(), Point(1, 2), HasherRegistry(), ignore_unhashable=True)
    assert len(result) == 64

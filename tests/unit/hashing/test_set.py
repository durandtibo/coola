from __future__ import annotations

import pytest

from coola.hashing import HasherRegistry, SetHasher, StrHasher, get_default_registry
from tests.unit.hashing.helpers import hash_with


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: StrHasher()})


###############################
#     Tests for SetHasher     #
###############################


def test_set_hasher_repr() -> None:
    assert repr(SetHasher()) == "SetHasher(tag='set')"


def test_set_hasher_str() -> None:
    assert str(SetHasher()) == "SetHasher(tag=set)"


def test_set_hasher_hash_returns_str_of_requested_length_returns_str(
    registry: HasherRegistry,
) -> None:
    result = hash_with(SetHasher(), {1, 2, 3}, registry, length=16)
    assert isinstance(result, str)


def test_set_hasher_hash_returns_str_of_requested_length_requested_length(
    registry: HasherRegistry,
) -> None:
    result = hash_with(SetHasher(), {1, 2, 3}, registry, length=16)
    assert len(result) == 16


def test_set_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = SetHasher()
    assert hash_with(hasher, {1, 2, 3}, registry) == hash_with(hasher, {1, 2, 3}, registry)


def test_set_hasher_hash_is_independent_of_insertion_order(registry: HasherRegistry) -> None:
    hasher = SetHasher()
    items = ["a", "b", "c", "d"]
    first = hash_with(hasher, set(items), registry)
    second = hash_with(hasher, set(items[::-1]), registry)  # noqa: C415
    assert first == second


def test_set_hasher_hash_different_items_different_hashes(registry: HasherRegistry) -> None:
    hasher = SetHasher()
    assert hash_with(hasher, {1, 2}, registry) != hash_with(hasher, {1, 3}, registry)


def test_set_hasher_hash_empty_set_differs_from_non_empty(registry: HasherRegistry) -> None:
    hasher = SetHasher()
    assert hash_with(hasher, set(), registry) != hash_with(hasher, {1}, registry)


def test_set_hasher_hash_tag_changes_hash(registry: HasherRegistry) -> None:
    assert hash_with(SetHasher("set"), {1}, registry) != hash_with(
        SetHasher("frozenset"), {1}, registry
    )


def test_set_hasher_hash_items_are_not_required_to_be_orderable() -> None:
    registry = get_default_registry()
    assert registry.hash({1, "a", None}) == registry.hash({None, "a", 1})


def test_set_hasher_hash_unhashable_item_raises() -> None:
    with pytest.raises(KeyError, match="is not registered"):
        hash_with(SetHasher(), {1}, HasherRegistry())


def test_set_hasher_hash_nested_frozenset_in_set() -> None:
    registry = get_default_registry()
    assert registry.hash({frozenset({1, 2})}) == registry.hash({frozenset({2, 1})})

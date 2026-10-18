from __future__ import annotations

import pytest

from coola.hashing import HasherRegistry, SetHasher, StrHasher, get_default_registry


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: StrHasher()})


##############################
#     Tests for SetHasher    #
##############################


def test_set_hasher_repr() -> None:
    assert repr(SetHasher()) == "SetHasher(tag='set')"


def test_set_hasher_str() -> None:
    assert str(SetHasher()) == "SetHasher(tag=set)"


def test_set_hasher_hash_returns_str_of_requested_length(registry: HasherRegistry) -> None:
    result = SetHasher().hash({1, 2, 3}, registry=registry, length=16)
    assert isinstance(result, str)
    assert len(result) == 16


def test_set_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = SetHasher()
    assert hasher.hash({1, 2, 3}, registry=registry) == hasher.hash({1, 2, 3}, registry=registry)


def test_set_hasher_hash_is_independent_of_insertion_order(registry: HasherRegistry) -> None:
    hasher = SetHasher()
    first = hasher.hash(set([1, 2, 3]), registry=registry)
    second = hasher.hash(set([3, 2, 1]), registry=registry)
    assert first == second


def test_set_hasher_hash_different_items_different_hashes(registry: HasherRegistry) -> None:
    hasher = SetHasher()
    assert hasher.hash({1, 2}, registry=registry) != hasher.hash({1, 3}, registry=registry)


def test_set_hasher_hash_empty_set_differs_from_non_empty(registry: HasherRegistry) -> None:
    hasher = SetHasher()
    assert hasher.hash(set(), registry=registry) != hasher.hash({1}, registry=registry)


def test_set_hasher_hash_tag_changes_hash(registry: HasherRegistry) -> None:
    assert SetHasher("set").hash({1}, registry=registry) != SetHasher("frozenset").hash(
        {1}, registry=registry
    )


def test_set_hasher_hash_items_are_not_required_to_be_orderable() -> None:
    registry = get_default_registry()
    assert registry.hash({1, "a", None}) == registry.hash({None, "a", 1})


def test_set_hasher_hash_unhashable_item_raises(registry: HasherRegistry) -> None:
    empty = HasherRegistry()
    with pytest.raises(KeyError):
        SetHasher().hash({1}, registry=empty)


def test_set_hasher_hash_nested_frozenset_in_set() -> None:
    registry = get_default_registry()
    assert registry.hash({frozenset({1, 2})}) == registry.hash({frozenset({2, 1})})

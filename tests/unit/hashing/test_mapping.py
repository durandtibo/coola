from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from coola.hashing import HasherRegistry, MappingHasher, StrHasher
from tests.unit.hashing.helpers import Unhashable


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: StrHasher(), Mapping: MappingHasher()})


###################################
#     Tests for MappingHasher     #
###################################


def test_mapping_hasher_repr() -> None:
    assert repr(MappingHasher()) == "MappingHasher(tag='mapping', ordered=False)"


def test_mapping_hasher_str() -> None:
    assert str(MappingHasher()) == "MappingHasher(tag=mapping, ordered=False)"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            {"a": 1, "b": 2},
            "0fde8be56160bfbace2c2f3ce1200a3a0b1355d40defca6fd85d8a423099a90a",
            id="two_items",
        ),
        pytest.param(
            {"a": 1},
            "08d6ce551687c99a97404de46b5b5dcc53d00d08b28301cdc149163a34d32355",
            id="single_item",
        ),
        pytest.param(
            {},
            "d0184328135d1a4b3a9492b8d5a214b49e15af380c2458d8a2671ecef2d2ee34",
            id="empty_mapping",
        ),
        pytest.param(
            {"x": "hello", "y": [1, 2, 3]},
            "e205a3845cc143f8b78e8e845ef51f928bc7900cb2425b865e4d2777118c3fd5",
            id="mixed_value_types",
        ),
    ],
)
def test_mapping_hasher_hash_parametrized(
    data: Any, expected: str, registry: HasherRegistry
) -> None:
    assert MappingHasher().hash(data, registry=registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "83405ad8236ae11d", id="16"),
        pytest.param(32, "d3d84762c7ee1934100661166b93a1f1", id="32"),
        pytest.param(
            64, "0fde8be56160bfbace2c2f3ce1200a3a0b1355d40defca6fd85d8a423099a90a", id="64-default"
        ),
    ],
)
def test_mapping_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = MappingHasher().hash({"a": 1, "b": 2}, registry=registry, length=length)
    assert result == expected
    assert len(result) == length


def test_mapping_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(MappingHasher().hash({"a": 1}, registry=registry), str)


def test_mapping_hasher_hash_is_insertion_order_independent(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hasher.hash({"a": 1, "b": 2}, registry=registry) == hasher.hash(
        {"b": 2, "a": 1}, registry=registry
    )


def test_mapping_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hasher.hash({"a": 1, "b": 2}, registry=registry) == hasher.hash(
        {"a": 1, "b": 2}, registry=registry
    )


def test_mapping_hasher_hash_different_keys_different_hashes(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hasher.hash({"a": 1}, registry=registry) != hasher.hash({"b": 1}, registry=registry)


def test_mapping_hasher_hash_different_values_different_hashes(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hasher.hash({"a": 1}, registry=registry) != hasher.hash({"a": 2}, registry=registry)


def test_mapping_hasher_hash_key_value_not_commutative(registry: HasherRegistry) -> None:
    # {"a": "b"} and {"b": "a"} must produce different hashes since key and
    # value hashes are concatenated in a fixed key-then-value order.
    hasher = MappingHasher()
    assert hasher.hash({"a": "b"}, registry=registry) != hasher.hash({"b": "a"}, registry=registry)


def test_mapping_hasher_hash_heterogeneous_key_types_does_not_raise(
    registry: HasherRegistry,
) -> None:
    # Keys of mutually incomparable types must not be sorted directly,
    # otherwise `sorted` raises `TypeError: '<' not supported between
    # instances of 'str' and 'int'`.
    result = MappingHasher().hash({1: "a", "b": 2}, registry=registry)
    assert isinstance(result, str)


def test_mapping_hasher_hash_heterogeneous_key_types_is_insertion_order_independent(
    registry: HasherRegistry,
) -> None:
    hasher = MappingHasher()
    assert hasher.hash({1: "a", "b": 2}, registry=registry) == hasher.hash(
        {"b": 2, 1: "a"}, registry=registry
    )


def test_mapping_hasher_hash_heterogeneous_key_types_is_deterministic(
    registry: HasherRegistry,
) -> None:
    hasher = MappingHasher()
    assert hasher.hash({1: "a", "b": 2}, registry=registry) == hasher.hash(
        {1: "a", "b": 2}, registry=registry
    )


@pytest.fixture
def registry_without_fallback() -> HasherRegistry:
    # No `object` fallback registered, so `Unhashable` cannot be resolved.
    return HasherRegistry({Mapping: MappingHasher(), dict: MappingHasher()})


def test_mapping_hasher_hash_unhashable_value_raises_by_default(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        MappingHasher().hash({"a": Unhashable()}, registry=registry_without_fallback)


def test_mapping_hasher_hash_unhashable_value_raises_when_explicitly_false(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        MappingHasher().hash(
            {"a": Unhashable()}, registry=registry_without_fallback, ignore_unhashable=False
        )


def test_mapping_hasher_hash_ignore_unhashable_value_returns_str(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        result = MappingHasher().hash(
            {"a": Unhashable()}, registry=registry_without_fallback, ignore_unhashable=True
        )
    assert isinstance(result, str)


def test_mapping_hasher_hash_ignore_unhashable_key_returns_str(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        result = MappingHasher().hash(
            {Unhashable(): "a"}, registry=registry_without_fallback, ignore_unhashable=True
        )
    assert isinstance(result, str)


def test_mapping_hasher_hash_ignore_unhashable_is_deterministic(
    registry_without_fallback: HasherRegistry,
) -> None:
    hasher = MappingHasher()
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        first = hasher.hash(
            {"a": Unhashable()}, registry=registry_without_fallback, ignore_unhashable=True
        )
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        second = hasher.hash(
            {"a": Unhashable()}, registry=registry_without_fallback, ignore_unhashable=True
        )
    assert first == second


def test_mapping_hasher_repr_ordered() -> None:
    assert repr(MappingHasher("OrderedDict", ordered=True)) == (
        "MappingHasher(tag='OrderedDict', ordered=True)"
    )


def test_mapping_hasher_hash_ordered_depends_on_insertion_order(registry: HasherRegistry) -> None:
    hasher = MappingHasher(ordered=True)
    assert hasher.hash({"a": 1, "b": 2}, registry=registry) != hasher.hash(
        {"b": 2, "a": 1}, registry=registry
    )


def test_mapping_hasher_hash_unordered_ignores_insertion_order(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hasher.hash({"a": 1, "b": 2}, registry=registry) == hasher.hash(
        {"b": 2, "a": 1}, registry=registry
    )


def test_mapping_hasher_tag_changes_hash(registry: HasherRegistry) -> None:
    assert MappingHasher("dict").hash({"a": 1}, registry=registry) != MappingHasher(
        "OrderedDict"
    ).hash({"a": 1}, registry=registry)

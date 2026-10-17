from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from coola.hashing import HasherRegistry, MappingHasher, StrHasher


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: StrHasher(), Mapping: MappingHasher()})


###################################
#     Tests for MappingHasher     #
###################################


def test_mapping_hasher_repr() -> None:
    assert repr(MappingHasher()) == "MappingHasher()"


def test_mapping_hasher_str() -> None:
    assert str(MappingHasher()) == "MappingHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            {"a": 1, "b": 2},
            "30dc27fd019ada1ebfd094059a3e349b601c452176d2c2fa5fce3c64678dc59b",
            id="two_items",
        ),
        pytest.param(
            {"a": 1},
            "35dc276c439c54e18dc675aaef88f84c1b198cd14811f51a95d092dc4603b959",
            id="single_item",
        ),
        pytest.param(
            {},
            "d0184328135d1a4b3a9492b8d5a214b49e15af380c2458d8a2671ecef2d2ee34",
            id="empty_mapping",
        ),
        pytest.param(
            {"x": "hello", "y": [1, 2, 3]},
            "10cdf21a84e7be45961a09182aa270f2408910513e75f3bc597d73fcb53ff02a",
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
        pytest.param(16, "b72dbd8958f6e2c8", id="16"),
        pytest.param(32, "9d58c48115b2aba85b3b75c6639775d0", id="32"),
        pytest.param(
            64, "30dc27fd019ada1ebfd094059a3e349b601c452176d2c2fa5fce3c64678dc59b", id="64-default"
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


class Unhashable:
    r"""A type with no registered hasher."""


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
    result = MappingHasher().hash(
        {"a": Unhashable()}, registry=registry_without_fallback, ignore_unhashable=True
    )
    assert isinstance(result, str)


def test_mapping_hasher_hash_ignore_unhashable_key_returns_str(
    registry_without_fallback: HasherRegistry,
) -> None:
    result = MappingHasher().hash(
        {Unhashable(): "a"}, registry=registry_without_fallback, ignore_unhashable=True
    )
    assert isinstance(result, str)


def test_mapping_hasher_hash_ignore_unhashable_is_deterministic(
    registry_without_fallback: HasherRegistry,
) -> None:
    hasher = MappingHasher()
    assert hasher.hash(
        {"a": Unhashable()}, registry=registry_without_fallback, ignore_unhashable=True
    ) == hasher.hash(
        {"a": Unhashable()}, registry=registry_without_fallback, ignore_unhashable=True
    )

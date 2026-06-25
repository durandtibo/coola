from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from coola.hashing import DefaultHasher, HasherRegistry, MappingHasher


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: DefaultHasher(), Mapping: MappingHasher()})


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
            "a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f",
            id="two_items",
        ),
        pytest.param(
            {"a": 1},
            "4ced54d7abe496efe275f1d45a9d4210be8ec6b4a1686dc8ce5d408b918a4089",
            id="single_item",
        ),
        pytest.param(
            {},
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8",
            id="empty_mapping",
        ),
        pytest.param(
            {"x": "hello", "y": [1, 2, 3]},
            "df791b222ef5922c2aaaf313d9df9d4de9f41e098d71890abdea038c26b45c9e",
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
        pytest.param(16, "11b35be24b54a602", id="16"),
        pytest.param(32, "6a2bc12783a1b90e6ddbe778ab44f452", id="32"),
        pytest.param(
            64, "a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f", id="64-default"
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

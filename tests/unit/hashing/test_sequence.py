from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

import pytest

from coola.hashing import HasherRegistry, SequenceHasher, StrHasher
from tests.unit.hashing.helpers import Unhashable, hash_with


class Point(NamedTuple):
    x: int
    y: int


class Renamed(NamedTuple):
    a: int
    b: int


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry(
        {
            object: StrHasher(),
            Sequence: SequenceHasher(),
            list: SequenceHasher(),
            tuple: SequenceHasher(),
        }
    )


####################################
#     Tests for SequenceHasher     #
####################################


def test_sequence_hasher_repr() -> None:
    assert repr(SequenceHasher()) == "SequenceHasher(tag='sequence')"


def test_sequence_hasher_str() -> None:
    assert str(SequenceHasher()) == "SequenceHasher(tag=sequence)"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            [1, 2, 3],
            "d4b06c73382157bf177e93c51702d837aa65bdde42ac5569528babb18797a387",
            id="list_of_ints",
        ),
        pytest.param(
            (1, 2, 3),
            "d4b06c73382157bf177e93c51702d837aa65bdde42ac5569528babb18797a387",
            id="tuple_default_tag",
        ),
        pytest.param(
            [42],
            "a68836d312928064e309ac695890de38ed7e84404ca0e50254fc414b6d9af4c9",
            id="single_element",
        ),
        pytest.param(
            [],
            "a22c70eed7e818075d623bbe39c012daef9b7450cca490bb1c48bb92a53fbac0",
            id="empty_sequence",
        ),
    ],
)
def test_sequence_hasher_hash_parametrized(
    data: Any, expected: str, registry: HasherRegistry
) -> None:
    assert hash_with(SequenceHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "d4b06c73382157bf", id="16"),
        pytest.param(32, "d4b06c73382157bf177e93c51702d837", id="32"),
        pytest.param(
            64, "d4b06c73382157bf177e93c51702d837aa65bdde42ac5569528babb18797a387", id="64-default"
        ),
    ],
)
def test_sequence_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(SequenceHasher(), [1, 2, 3], registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_sequence_hasher_hash_length_length_is_respected(
    length: int, registry: HasherRegistry
) -> None:
    result = hash_with(SequenceHasher(), [1, 2, 3], registry, length=length)
    assert len(result) == length


def test_sequence_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(SequenceHasher(), [1, 2, 3], registry), str)


def test_sequence_hasher_hash_list_and_tuple_produce_same_hash(
    registry: HasherRegistry,
) -> None:
    hasher = SequenceHasher()
    assert hash_with(hasher, [1, 2, 3], registry) == hash_with(hasher, (1, 2, 3), registry)


def test_sequence_hasher_hash_is_order_sensitive(registry: HasherRegistry) -> None:
    hasher = SequenceHasher()
    assert hash_with(hasher, [1, 2, 3], registry) != hash_with(hasher, [3, 2, 1], registry)


def test_sequence_hasher_hash_is_length_sensitive(registry: HasherRegistry) -> None:
    hasher = SequenceHasher()
    assert hash_with(hasher, [1, 2], registry) != hash_with(hasher, [1, 2, 3], registry)


def test_sequence_hasher_hash_different_values_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = SequenceHasher()
    assert hash_with(hasher, [1, 2, 3], registry) != hash_with(hasher, [4, 5, 6], registry)


def test_sequence_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = SequenceHasher()
    assert hash_with(hasher, [1, 2, 3], registry) == hash_with(hasher, [1, 2, 3], registry)


def test_sequence_hasher_hash_nested_sequence(registry: HasherRegistry) -> None:
    # Nested lists are recursed into via the registry.
    assert hash_with(SequenceHasher(), [[1, 2], [3, 4]], registry) == (
        "2d6a9e7791286108a2527f96ca71fb2e8eae0ec4fc958f2ade02fbfddba453dd"
    )


def test_sequence_hasher_hash_nested_differs_from_flat(registry: HasherRegistry) -> None:
    # [[1, 2], [3, 4]] should not hash the same as [1, 2, 3, 4].
    hasher = SequenceHasher()
    assert hash_with(hasher, [[1, 2], [3, 4]], registry) != hash_with(
        hasher, [1, 2, 3, 4], registry
    )


@pytest.fixture
def registry_without_fallback() -> HasherRegistry:
    # No `object` fallback registered, so `Unhashable` cannot be resolved.
    return HasherRegistry({list: SequenceHasher()})


def test_sequence_hasher_hash_unhashable_item_raises_by_default(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_with(SequenceHasher(), [1, Unhashable(), 3], registry_without_fallback)


def test_sequence_hasher_hash_unhashable_item_raises_when_explicitly_false(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_with(
            SequenceHasher(),
            [1, Unhashable(), 3],
            registry_without_fallback,
            ignore_unhashable=False,
        )


def test_sequence_hasher_hash_ignore_unhashable_item_returns_str_warns(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        hash_with(
            SequenceHasher(),
            [1, Unhashable(), 3],
            registry_without_fallback,
            ignore_unhashable=True,
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sequence_hasher_hash_ignore_unhashable_item_returns_str(
    registry_without_fallback: HasherRegistry,
) -> None:
    result = hash_with(
        SequenceHasher(), [1, Unhashable(), 3], registry_without_fallback, ignore_unhashable=True
    )
    assert isinstance(result, str)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sequence_hasher_hash_ignore_unhashable_is_deterministic(
    registry_without_fallback: HasherRegistry,
) -> None:
    hasher = SequenceHasher()
    first = hash_with(
        hasher, [1, Unhashable(), 3], registry_without_fallback, ignore_unhashable=True
    )
    second = hash_with(
        hasher, [1, Unhashable(), 3], registry_without_fallback, ignore_unhashable=True
    )
    assert first == second


def test_sequence_hasher_tag_changes_hash(registry: HasherRegistry) -> None:
    assert hash_with(SequenceHasher("list"), [1, 2], registry) != hash_with(
        SequenceHasher("tuple"), [1, 2], registry
    )


def test_sequence_hasher_repr_custom_tag() -> None:
    assert repr(SequenceHasher("list")) == "SequenceHasher(tag='list')"


def test_sequence_hasher_hash_named_tuple_differs_from_tuple(registry: HasherRegistry) -> None:
    hasher = SequenceHasher("tuple")
    assert hash_with(hasher, Point(1, 2), registry) != hash_with(hasher, (1, 2), registry)


def test_sequence_hasher_hash_named_tuple_depends_on_field_names(
    registry: HasherRegistry,
) -> None:
    hasher = SequenceHasher("tuple")
    assert hash_with(hasher, Point(1, 2), registry) != hash_with(hasher, Renamed(1, 2), registry)

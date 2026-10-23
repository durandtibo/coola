from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple
from unittest.mock import Mock

import pytest

from coola.hashing import HasherRegistry, SequenceHasher, StrHasher
from tests.unit.hashing.helpers import Unhashable


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
            "bfe86f0fc5750a217446b5a88d2bfc543b33c63fa190489e917b16a0bc792cce",
            id="list_of_ints",
        ),
        pytest.param(
            (1, 2, 3),
            "bfe86f0fc5750a217446b5a88d2bfc543b33c63fa190489e917b16a0bc792cce",
            id="tuple_default_tag",
        ),
        pytest.param(
            [42],
            "973d92e388edf5f5952cf69bb7d85b68114ee8fb906c5cfbdbc48a8b265b4909",
            id="single_element",
        ),
        pytest.param(
            [],
            "844dea01562659240f7c438c4bc74c7713eb68c6e4d693a30bda1798aaa972d7",
            id="empty_sequence",
        ),
        pytest.param(
            "abc",
            "aa15256e3813f16a427f4796612909858040b64185401a6a65c05c668c19b029",
            id="str_sequence",
        ),
    ],
)
def test_sequence_hasher_hash_parametrized(
    data: Any, expected: str, registry: HasherRegistry
) -> None:
    assert SequenceHasher().hash(data, registry=registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "b41e8f79a38a42d1", id="16"),
        pytest.param(32, "65b29e2995d9414e93e4599d87c72219", id="32"),
        pytest.param(
            64, "bfe86f0fc5750a217446b5a88d2bfc543b33c63fa190489e917b16a0bc792cce", id="64-default"
        ),
    ],
)
def test_sequence_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = SequenceHasher().hash([1, 2, 3], registry=registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_sequence_hasher_hash_length_length_is_respected(
    length: int, registry: HasherRegistry
) -> None:
    result = SequenceHasher().hash([1, 2, 3], registry=registry, length=length)
    assert len(result) == length


def test_sequence_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(SequenceHasher().hash([1, 2, 3], registry=registry), str)


def test_sequence_hasher_hash_list_and_tuple_produce_same_hash(
    registry: HasherRegistry,
) -> None:
    hasher = SequenceHasher()
    assert hasher.hash([1, 2, 3], registry=registry) == hasher.hash((1, 2, 3), registry=registry)


def test_sequence_hasher_hash_is_order_sensitive(registry: HasherRegistry) -> None:
    hasher = SequenceHasher()
    assert hasher.hash([1, 2, 3], registry=registry) != hasher.hash([3, 2, 1], registry=registry)


def test_sequence_hasher_hash_is_length_sensitive(registry: HasherRegistry) -> None:
    hasher = SequenceHasher()
    assert hasher.hash([1, 2], registry=registry) != hasher.hash([1, 2, 3], registry=registry)


def test_sequence_hasher_hash_different_values_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = SequenceHasher()
    assert hasher.hash([1, 2, 3], registry=registry) != hasher.hash([4, 5, 6], registry=registry)


def test_sequence_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = SequenceHasher()
    assert hasher.hash([1, 2, 3], registry=registry) == hasher.hash([1, 2, 3], registry=registry)


def test_sequence_hasher_hash_nested_sequence(registry: HasherRegistry) -> None:
    # Nested lists are recursed into via the registry.
    assert SequenceHasher().hash([[1, 2], [3, 4]], registry=registry) == (
        "7b31321709b3a28e0209236aae8fb5a69ab823d68b35660ff88459e60409c63f"
    )


def test_sequence_hasher_hash_nested_differs_from_flat(registry: HasherRegistry) -> None:
    # [[1, 2], [3, 4]] should not hash the same as [1, 2, 3, 4].
    hasher = SequenceHasher()
    assert hasher.hash([[1, 2], [3, 4]], registry=registry) != hasher.hash(
        [1, 2, 3, 4], registry=registry
    )


@pytest.fixture
def registry_without_fallback() -> HasherRegistry:
    # No `object` fallback registered, so `Unhashable` cannot be resolved.
    return HasherRegistry({list: SequenceHasher()})


def test_sequence_hasher_hash_unhashable_item_raises_by_default(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        SequenceHasher().hash([1, Unhashable(), 3], registry=registry_without_fallback)


def test_sequence_hasher_hash_unhashable_item_raises_when_explicitly_false(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        SequenceHasher().hash(
            [1, Unhashable(), 3], registry=registry_without_fallback, ignore_unhashable=False
        )


def test_sequence_hasher_hash_ignore_unhashable_item_returns_str_warns(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        SequenceHasher().hash(
            [1, Unhashable(), 3], registry=registry_without_fallback, ignore_unhashable=True
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sequence_hasher_hash_ignore_unhashable_item_returns_str(
    registry_without_fallback: HasherRegistry,
) -> None:
    result = SequenceHasher().hash(
        [1, Unhashable(), 3], registry=registry_without_fallback, ignore_unhashable=True
    )
    assert isinstance(result, str)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sequence_hasher_hash_ignore_unhashable_is_deterministic(
    registry_without_fallback: HasherRegistry,
) -> None:
    hasher = SequenceHasher()
    first = hasher.hash(
        [1, Unhashable(), 3], registry=registry_without_fallback, ignore_unhashable=True
    )
    second = hasher.hash(
        [1, Unhashable(), 3], registry=registry_without_fallback, ignore_unhashable=True
    )
    assert first == second


def test_sequence_hasher_hash_does_not_reuse_cache_across_different_objects(
    registry: HasherRegistry,
) -> None:
    wrapped = Mock(wraps=registry)
    SequenceHasher().hash([1, 2, 3], registry=wrapped)
    assert wrapped.hash.call_count == 3


def test_sequence_hasher_tag_changes_hash(registry: HasherRegistry) -> None:
    assert SequenceHasher("list").hash([1, 2], registry=registry) != SequenceHasher("tuple").hash(
        [1, 2], registry=registry
    )


def test_sequence_hasher_repr_custom_tag() -> None:
    assert repr(SequenceHasher("list")) == "SequenceHasher(tag='list')"


def test_sequence_hasher_hash_named_tuple_differs_from_tuple(registry: HasherRegistry) -> None:
    hasher = SequenceHasher("tuple")
    assert hasher.hash(Point(1, 2), registry=registry) != hasher.hash((1, 2), registry=registry)


def test_sequence_hasher_hash_named_tuple_depends_on_field_names(
    registry: HasherRegistry,
) -> None:
    hasher = SequenceHasher("tuple")
    assert hasher.hash(Point(1, 2), registry=registry) != hasher.hash(
        Renamed(1, 2), registry=registry
    )

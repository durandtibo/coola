from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from coola.hashing import HasherRegistry, SequenceHasher, StrHasher


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
    assert repr(SequenceHasher()) == "SequenceHasher()"


def test_sequence_hasher_str() -> None:
    assert str(SequenceHasher()) == "SequenceHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            [1, 2, 3],
            "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de",
            id="list_of_ints",
        ),
        pytest.param(
            (1, 2, 3),
            "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de",
            id="tuple_same_as_list",
        ),
        pytest.param(
            [42],
            "52cd1083fe122c990268e9a127b49f5dc65a0b071a3a7c00f52f01b2208c1a38",
            id="single_element",
        ),
        pytest.param(
            [],
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8",
            id="empty_sequence",
        ),
        pytest.param(
            "abc",
            "040c910c2fffa8ec50282e64abfd994192e99a5565b07855bbcc23669bb6f25a",
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
        pytest.param(16, "38600139ffd04ca0", id="16"),
        pytest.param(32, "56ef471038da2edee2635fd3064628a1", id="32"),
        pytest.param(
            64, "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de", id="64-default"
        ),
    ],
)
def test_sequence_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = SequenceHasher().hash([1, 2, 3], registry=registry, length=length)
    assert result == expected
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
        "aaaf7ff74d087a45da40168d0b346ceb24a7e08793a9001390e10103d5736ccd"
    )


def test_sequence_hasher_hash_nested_differs_from_flat(registry: HasherRegistry) -> None:
    # [[1, 2], [3, 4]] should not hash the same as [1, 2, 3, 4].
    hasher = SequenceHasher()
    assert hasher.hash([[1, 2], [3, 4]], registry=registry) != hasher.hash(
        [1, 2, 3, 4], registry=registry
    )

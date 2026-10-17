from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import Mock

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
            "0708bf79f681139268711c8c0f125fde9109476dd1362adc3c3ce455e605b94e",
            id="list_of_ints",
        ),
        pytest.param(
            (1, 2, 3),
            "0708bf79f681139268711c8c0f125fde9109476dd1362adc3c3ce455e605b94e",
            id="tuple_same_as_list",
        ),
        pytest.param(
            [42],
            "b989e403fcb0a92fbd9cdd8e83b24c8540f4632db582a02dda087916e1ec7cca",
            id="single_element",
        ),
        pytest.param(
            [],
            "844dea01562659240f7c438c4bc74c7713eb68c6e4d693a30bda1798aaa972d7",
            id="empty_sequence",
        ),
        pytest.param(
            "abc",
            "deeb2844f1898c0f1fd0107f79100f0f4458f597ce5ba3082cf51d19c628275e",
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
        pytest.param(16, "c707ce89f1b1a4f6", id="16"),
        pytest.param(32, "df4ebc44992eea89c78876774c07e0df", id="32"),
        pytest.param(
            64, "0708bf79f681139268711c8c0f125fde9109476dd1362adc3c3ce455e605b94e", id="64-default"
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
        "b47a1417d5bb6b7e1ad2e4c3e7bd5f2c06676b46e1529deb24d1d638c8e64cc4"
    )


def test_sequence_hasher_hash_nested_differs_from_flat(registry: HasherRegistry) -> None:
    # [[1, 2], [3, 4]] should not hash the same as [1, 2, 3, 4].
    hasher = SequenceHasher()
    assert hasher.hash([[1, 2], [3, 4]], registry=registry) != hasher.hash(
        [1, 2, 3, 4], registry=registry
    )


class Unhashable:
    r"""A type with no registered hasher."""


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


def test_sequence_hasher_hash_ignore_unhashable_item_returns_str(
    registry_without_fallback: HasherRegistry,
) -> None:
    result = SequenceHasher().hash(
        [1, Unhashable(), 3], registry=registry_without_fallback, ignore_unhashable=True
    )
    assert isinstance(result, str)


def test_sequence_hasher_hash_ignore_unhashable_is_deterministic(
    registry_without_fallback: HasherRegistry,
) -> None:
    hasher = SequenceHasher()
    assert hasher.hash(
        [1, Unhashable(), 3], registry=registry_without_fallback, ignore_unhashable=True
    ) == hasher.hash(
        [1, Unhashable(), 3], registry=registry_without_fallback, ignore_unhashable=True
    )


def test_sequence_hasher_hash_does_not_reuse_cache_across_different_objects(
    registry: HasherRegistry,
) -> None:
    wrapped = Mock(wraps=registry)
    SequenceHasher().hash([1, 2, 3], registry=wrapped)
    assert wrapped.hash.call_count == 3

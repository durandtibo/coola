from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from coola.hashing import HasherRegistry, MappingHasher, StrHasher
from tests.unit.hashing.helpers import Unhashable, hash_with


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
            "e6952362146927fc104d6be3f2c359d55a545e263b71d1feac43b2625e2dcf48",
            id="two_items",
        ),
        pytest.param(
            {"a": 1},
            "dad1237221ba0de64c4636d351158401530fff49d6d952a02b7155af0637233a",
            id="single_item",
        ),
        pytest.param(
            {},
            "6662e425fd8dcb2032f333732915004156c02bc14bcacf74f8c311f80a732b34",
            id="empty_mapping",
        ),
        pytest.param(
            {"x": "hello", "y": [1, 2, 3]},
            "163f5841d925f3cfa9035950d5d87535326209063b75e66b755cddb38077f793",
            id="mixed_value_types",
        ),
    ],
)
def test_mapping_hasher_hash_parametrized(
    data: Any, expected: str, registry: HasherRegistry
) -> None:
    assert hash_with(MappingHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "e6952362146927fc", id="16"),
        pytest.param(32, "e6952362146927fc104d6be3f2c359d5", id="32"),
        pytest.param(
            64, "e6952362146927fc104d6be3f2c359d55a545e263b71d1feac43b2625e2dcf48", id="64-default"
        ),
    ],
)
def test_mapping_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(MappingHasher(), {"a": 1, "b": 2}, registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_mapping_hasher_hash_length_is_respected(length: int, registry: HasherRegistry) -> None:
    result = hash_with(MappingHasher(), {"a": 1, "b": 2}, registry, length=length)
    assert len(result) == length


def test_mapping_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(MappingHasher(), {"a": 1}, registry), str)


def test_mapping_hasher_hash_is_insertion_order_independent(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hash_with(hasher, {"a": 1, "b": 2}, registry) == hash_with(
        hasher, {"b": 2, "a": 1}, registry
    )


def test_mapping_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hash_with(hasher, {"a": 1, "b": 2}, registry) == hash_with(
        hasher, {"a": 1, "b": 2}, registry
    )


def test_mapping_hasher_hash_different_keys_different_hashes(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hash_with(hasher, {"a": 1}, registry) != hash_with(hasher, {"b": 1}, registry)


def test_mapping_hasher_hash_different_values_different_hashes(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hash_with(hasher, {"a": 1}, registry) != hash_with(hasher, {"a": 2}, registry)


def test_mapping_hasher_hash_key_value_not_commutative(registry: HasherRegistry) -> None:
    # {"a": "b"} and {"b": "a"} must produce different hashes since key and
    # value hashes are concatenated in a fixed key-then-value order.
    hasher = MappingHasher()
    assert hash_with(hasher, {"a": "b"}, registry) != hash_with(hasher, {"b": "a"}, registry)


def test_mapping_hasher_hash_heterogeneous_key_types_does_not_raise(
    registry: HasherRegistry,
) -> None:
    # Keys of mutually incomparable types must not be sorted directly,
    # otherwise `sorted` raises `TypeError: '<' not supported between
    # instances of 'str' and 'int'`.
    result = hash_with(MappingHasher(), {1: "a", "b": 2}, registry)
    assert isinstance(result, str)


def test_mapping_hasher_hash_heterogeneous_key_types_is_insertion_order_independent(
    registry: HasherRegistry,
) -> None:
    hasher = MappingHasher()
    assert hash_with(hasher, {1: "a", "b": 2}, registry) == hash_with(
        hasher, {"b": 2, 1: "a"}, registry
    )


def test_mapping_hasher_hash_heterogeneous_key_types_is_deterministic(
    registry: HasherRegistry,
) -> None:
    hasher = MappingHasher()
    assert hash_with(hasher, {1: "a", "b": 2}, registry) == hash_with(
        hasher, {1: "a", "b": 2}, registry
    )


@pytest.fixture
def registry_without_fallback() -> HasherRegistry:
    # No `object` fallback registered, so `Unhashable` cannot be resolved.
    return HasherRegistry({Mapping: MappingHasher(), dict: MappingHasher()})


def test_mapping_hasher_hash_unhashable_value_raises_by_default(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_with(MappingHasher(), {"a": Unhashable()}, registry_without_fallback)


def test_mapping_hasher_hash_unhashable_value_raises_when_explicitly_false(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_with(
            MappingHasher(), {"a": Unhashable()}, registry_without_fallback, ignore_unhashable=False
        )


def test_mapping_hasher_hash_ignore_unhashable_value_returns_str_warns(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        hash_with(
            MappingHasher(), {"a": Unhashable()}, registry_without_fallback, ignore_unhashable=True
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mapping_hasher_hash_ignore_unhashable_value_returns_str(
    registry_without_fallback: HasherRegistry,
) -> None:
    result = hash_with(
        MappingHasher(), {"a": Unhashable()}, registry_without_fallback, ignore_unhashable=True
    )
    assert isinstance(result, str)


def test_mapping_hasher_hash_ignore_unhashable_key_returns_str_warns(
    registry_without_fallback: HasherRegistry,
) -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        hash_with(
            MappingHasher(), {Unhashable(): "a"}, registry_without_fallback, ignore_unhashable=True
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mapping_hasher_hash_ignore_unhashable_key_returns_str(
    registry_without_fallback: HasherRegistry,
) -> None:
    result = hash_with(
        MappingHasher(), {Unhashable(): "a"}, registry_without_fallback, ignore_unhashable=True
    )
    assert isinstance(result, str)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mapping_hasher_hash_ignore_unhashable_is_deterministic(
    registry_without_fallback: HasherRegistry,
) -> None:
    hasher = MappingHasher()
    first = hash_with(
        hasher, {"a": Unhashable()}, registry_without_fallback, ignore_unhashable=True
    )
    second = hash_with(
        hasher, {"a": Unhashable()}, registry_without_fallback, ignore_unhashable=True
    )
    assert first == second


def test_mapping_hasher_repr_ordered() -> None:
    assert repr(MappingHasher("OrderedDict", ordered=True)) == (
        "MappingHasher(tag='OrderedDict', ordered=True)"
    )


def test_mapping_hasher_hash_ordered_depends_on_insertion_order(registry: HasherRegistry) -> None:
    hasher = MappingHasher(ordered=True)
    assert hash_with(hasher, {"a": 1, "b": 2}, registry) != hash_with(
        hasher, {"b": 2, "a": 1}, registry
    )


def test_mapping_hasher_hash_unordered_ignores_insertion_order(registry: HasherRegistry) -> None:
    hasher = MappingHasher()
    assert hash_with(hasher, {"a": 1, "b": 2}, registry) == hash_with(
        hasher, {"b": 2, "a": 1}, registry
    )


def test_mapping_hasher_tag_changes_hash(registry: HasherRegistry) -> None:
    assert hash_with(MappingHasher("dict"), {"a": 1}, registry) != hash_with(
        MappingHasher("OrderedDict"), {"a": 1}, registry
    )


@pytest.mark.filterwarnings("ignore:No hasher registered:UserWarning")
def test_mapping_hasher_hash_keys_with_the_same_hash_do_not_depend_on_the_order() -> None:
    # Unhashable keys get the same placeholder hash with ignore_unhashable.
    registry = HasherRegistry({str: StrHasher(), Mapping: MappingHasher()})
    first = {Unhashable(): "a", Unhashable(): "b"}
    second = dict(reversed(list(first.items())))
    hasher = MappingHasher()
    assert hash_with(hasher, first, registry, ignore_unhashable=True) == hash_with(
        hasher, second, registry, ignore_unhashable=True
    )

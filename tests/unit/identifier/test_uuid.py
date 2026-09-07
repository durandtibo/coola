from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

import pytest

from coola.hashing import (
    HasherRegistry,
    ReprHasher,
    StringHasher,
    get_default_registry,
    hash_object,
)
from coola.identifier.uuid import _NAMESPACE, stable_uuid

if TYPE_CHECKING:
    from collections.abc import Generator

UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$")


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the singleton registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


##################################
#     Tests for stable_uuid      #
##################################


def test_stable_uuid_returns_str() -> None:
    assert isinstance(stable_uuid({"key": "value"}), str)


def test_stable_uuid_is_valid_uuid_v5() -> None:
    assert UUID_PATTERN.match(stable_uuid({"key": "value"}))


def test_stable_uuid_is_deterministic() -> None:
    data = {"source": "cats.txt", "page": 1}
    assert stable_uuid(data) == stable_uuid(data)


def test_stable_uuid_equal_dicts_same_uuid() -> None:
    assert stable_uuid({"source": "cats.txt", "page": 1}) == stable_uuid(
        {"source": "cats.txt", "page": 1}
    )


def test_stable_uuid_dict_key_order_independent() -> None:
    assert stable_uuid({"source": "cats.txt", "page": 1}) == stable_uuid(
        {"page": 1, "source": "cats.txt"}
    )


def test_stable_uuid_different_values_different_uuid() -> None:
    assert stable_uuid({"source": "cats.txt"}) != stable_uuid({"source": "dogs.txt"})


def test_stable_uuid_different_keys_different_uuid() -> None:
    assert stable_uuid({"source": "cats.txt"}) != stable_uuid({"origin": "cats.txt"})


def test_stable_uuid_empty_dict() -> None:
    assert UUID_PATTERN.match(stable_uuid({}))


def test_stable_uuid_empty_dict_same_uuid() -> None:
    assert stable_uuid({}) == stable_uuid({})


def test_stable_uuid_nested_dict_key_order_independent() -> None:
    assert stable_uuid({"info": {"year": 2024, "topic": "cats"}}) == stable_uuid(
        {"info": {"topic": "cats", "year": 2024}}
    )


def test_stable_uuid_list_input() -> None:
    assert UUID_PATTERN.match(stable_uuid([1, 2, 3]))


def test_stable_uuid_scalar_input() -> None:
    assert UUID_PATTERN.match(stable_uuid(42))


def test_stable_uuid_tuple_and_list_same_uuid() -> None:
    # SequenceHasher (used by the default registry for both list and
    # tuple) hashes items the same way regardless of the concrete
    # sequence type, so a list and an equivalent tuple derive the same
    # UUID.
    assert stable_uuid([1, 2]) == stable_uuid((1, 2))


def test_stable_uuid_matches_uuid5_of_hash_object() -> None:
    # stable_uuid is defined as uuid5 of hash_object's output under a
    # fixed namespace - this pins that relationship so the two helpers
    # cannot silently drift apart from each other.
    data = {"source": "cats.txt", "page": 1}
    assert stable_uuid(data) == str(uuid.uuid5(_NAMESPACE, hash_object(data)))


def test_stable_uuid_unhashable_type_raises_by_default() -> None:
    with pytest.raises(KeyError):
        stable_uuid(object())


def test_stable_uuid_ignore_unhashable_does_not_raise() -> None:
    assert UUID_PATTERN.match(stable_uuid(object(), ignore_unhashable=True))


def test_stable_uuid_ignore_unhashable_is_deterministic() -> None:
    assert stable_uuid(object(), ignore_unhashable=True) == stable_uuid(
        object(), ignore_unhashable=True
    )


def test_stable_uuid_uses_given_registry() -> None:
    registry = HasherRegistry({object: StringHasher()})
    assert stable_uuid("meow", registry=registry) == stable_uuid("meow", registry=registry)


def test_stable_uuid_custom_registry_is_honored() -> None:
    # The default registry hashes strings via StringHasher, character
    # content only; ReprHasher hashes the repr() form instead, so
    # routing all types through it changes the resulting digest (and
    # therefore the UUID) for the same input string.
    custom_registry = HasherRegistry({object: ReprHasher()})
    assert stable_uuid("meow") != stable_uuid("meow", registry=custom_registry)


def test_stable_uuid_does_not_mutate_default_registry_when_custom_registry_given() -> None:
    custom_registry = HasherRegistry({object: StringHasher()})
    stable_uuid("meow", registry=custom_registry)
    assert not hasattr(get_default_registry, "_registry")


def test_stable_uuid_custom_namespace_changes_uuid() -> None:
    other_namespace = uuid.uuid4()
    assert stable_uuid({"a": 1}) != stable_uuid({"a": 1}, namespace=other_namespace)


def test_stable_uuid_custom_namespace_is_deterministic() -> None:
    namespace = uuid.uuid4()
    data = {"a": 1}
    assert stable_uuid(data, namespace=namespace) == stable_uuid(data, namespace=namespace)


def test_stable_uuid_same_data_different_namespace_matches_manual_uuid5() -> None:
    namespace = uuid.uuid4()
    data = {"a": 1}
    assert stable_uuid(data, namespace=namespace) == str(uuid.uuid5(namespace, hash_object(data)))

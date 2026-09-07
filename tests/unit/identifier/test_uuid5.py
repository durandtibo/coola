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
from coola.identifier.uuid5 import _NAMESPACE, generate_stable_uuid5

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


############################################
#     Tests for generate_stable_uuid5      #
############################################


def test_generate_stable_uuid5_returns_str() -> None:
    assert isinstance(generate_stable_uuid5({"key": "value"}), str)


def test_generate_stable_uuid5_is_valid_uuid_v5() -> None:
    assert UUID_PATTERN.match(generate_stable_uuid5({"key": "value"}))


def test_generate_stable_uuid5_is_deterministic() -> None:
    data = {"source": "cats.txt", "page": 1}
    assert generate_stable_uuid5(data) == generate_stable_uuid5(data)


def test_generate_stable_uuid5_equal_dicts_same_uuid() -> None:
    assert generate_stable_uuid5({"source": "cats.txt", "page": 1}) == generate_stable_uuid5(
        {"source": "cats.txt", "page": 1}
    )


def test_generate_stable_uuid5_dict_key_order_independent() -> None:
    assert generate_stable_uuid5({"source": "cats.txt", "page": 1}) == generate_stable_uuid5(
        {"page": 1, "source": "cats.txt"}
    )


def test_generate_stable_uuid5_different_values_different_uuid() -> None:
    assert generate_stable_uuid5({"source": "cats.txt"}) != generate_stable_uuid5(
        {"source": "dogs.txt"}
    )


def test_generate_stable_uuid5_different_keys_different_uuid() -> None:
    assert generate_stable_uuid5({"source": "cats.txt"}) != generate_stable_uuid5(
        {"origin": "cats.txt"}
    )


def test_generate_stable_uuid5_empty_dict() -> None:
    assert UUID_PATTERN.match(generate_stable_uuid5({}))


def test_generate_stable_uuid5_empty_dict_same_uuid() -> None:
    assert generate_stable_uuid5({}) == generate_stable_uuid5({})


def test_generate_stable_uuid5_nested_dict_key_order_independent() -> None:
    assert generate_stable_uuid5({"info": {"year": 2024, "topic": "cats"}}) == generate_stable_uuid5(
        {"info": {"topic": "cats", "year": 2024}}
    )


def test_generate_stable_uuid5_list_input() -> None:
    assert UUID_PATTERN.match(generate_stable_uuid5([1, 2, 3]))


def test_generate_stable_uuid5_scalar_input() -> None:
    assert UUID_PATTERN.match(generate_stable_uuid5(42))


def test_generate_stable_uuid5_tuple_and_list_same_uuid() -> None:
    # SequenceHasher (used by the default registry for both list and
    # tuple) hashes items the same way regardless of the concrete
    # sequence type, so a list and an equivalent tuple derive the same
    # UUID.
    assert generate_stable_uuid5([1, 2]) == generate_stable_uuid5((1, 2))


def test_generate_stable_uuid5_matches_uuid5_of_hash_object() -> None:
    # generate_stable_uuid5 is defined as uuid5 of hash_object's output
    # under a fixed namespace - this pins that relationship so the two
    # helpers cannot silently drift apart from each other.
    data = {"source": "cats.txt", "page": 1}
    assert generate_stable_uuid5(data) == str(uuid.uuid5(_NAMESPACE, hash_object(data, length=128)))


def test_generate_stable_uuid5_unhashable_type_raises_by_default() -> None:
    with pytest.raises(KeyError):
        generate_stable_uuid5(object())


def test_generate_stable_uuid5_ignore_unhashable_does_not_raise() -> None:
    assert UUID_PATTERN.match(generate_stable_uuid5(object(), ignore_unhashable=True))


def test_generate_stable_uuid5_ignore_unhashable_is_deterministic() -> None:
    assert generate_stable_uuid5(object(), ignore_unhashable=True) == generate_stable_uuid5(
        object(), ignore_unhashable=True
    )


def test_generate_stable_uuid5_uses_given_registry() -> None:
    registry = HasherRegistry({object: StringHasher()})
    assert generate_stable_uuid5("meow", registry=registry) == generate_stable_uuid5(
        "meow", registry=registry
    )


def test_generate_stable_uuid5_custom_registry_is_honored() -> None:
    # The default registry hashes strings via StringHasher, character
    # content only; ReprHasher hashes the repr() form instead, so
    # routing all types through it changes the resulting digest (and
    # therefore the UUID) for the same input string.
    custom_registry = HasherRegistry({object: ReprHasher()})
    assert generate_stable_uuid5("meow") != generate_stable_uuid5("meow", registry=custom_registry)


def test_generate_stable_uuid5_does_not_mutate_default_registry_when_custom_registry_given() -> None:
    custom_registry = HasherRegistry({object: StringHasher()})
    generate_stable_uuid5("meow", registry=custom_registry)
    assert not hasattr(get_default_registry, "_registry")


def test_generate_stable_uuid5_custom_namespace_changes_uuid() -> None:
    other_namespace = uuid.uuid4()
    assert generate_stable_uuid5({"a": 1}) != generate_stable_uuid5(
        {"a": 1}, namespace=other_namespace
    )


def test_generate_stable_uuid5_custom_namespace_is_deterministic() -> None:
    namespace = uuid.uuid4()
    data = {"a": 1}
    assert generate_stable_uuid5(data, namespace=namespace) == generate_stable_uuid5(
        data, namespace=namespace
    )


def test_generate_stable_uuid5_same_data_different_namespace_matches_manual_uuid5() -> None:
    namespace = uuid.uuid4()
    data = {"a": 1}
    assert generate_stable_uuid5(data, namespace=namespace) == str(
        uuid.uuid5(namespace, hash_object(data, length=128))
    )

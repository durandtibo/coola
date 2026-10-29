from __future__ import annotations

import re
import uuid

import pytest

from coola.hashing import (
    HasherRegistry,
    ReprHasher,
    StringHasher,
    hash_object,
)
from coola.identifier.content import _NAMESPACE, generate_stable_content_id, generate_stable_uuid5

HEX_PATTERN = re.compile(r"^[0-9a-f]+$")
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$")


################################################
#     Tests for generate_stable_content_id     #
################################################


def test_generate_stable_content_id_returns_str() -> None:
    assert isinstance(generate_stable_content_id({"key": "value"}), str)


def test_generate_stable_content_id_default_length() -> None:
    assert len(generate_stable_content_id({"key": "value"})) == 64


def test_generate_stable_content_id_is_hex_string() -> None:
    assert HEX_PATTERN.match(generate_stable_content_id({"key": "value"}))


def test_generate_stable_content_id_custom_length() -> None:
    assert len(generate_stable_content_id({"key": "value"}, length=16)) == 16


def test_generate_stable_content_id_is_deterministic() -> None:
    data = {"source": "cats.txt", "page": 1}
    assert generate_stable_content_id(data) == generate_stable_content_id(data)


def test_generate_stable_content_id_dict_key_order_independent() -> None:
    assert generate_stable_content_id(
        {"source": "cats.txt", "page": 1}
    ) == generate_stable_content_id({"page": 1, "source": "cats.txt"})


def test_generate_stable_content_id_different_values_different_id() -> None:
    assert generate_stable_content_id({"source": "cats.txt"}) != generate_stable_content_id(
        {"source": "dogs.txt"}
    )


def test_generate_stable_content_id_empty_dict() -> None:
    assert HEX_PATTERN.match(generate_stable_content_id({}))


def test_generate_stable_content_id_list_input() -> None:
    assert HEX_PATTERN.match(generate_stable_content_id([1, 2, 3]))


def test_generate_stable_content_id_tuple_and_list_different_id() -> None:
    assert generate_stable_content_id([1, 2]) != generate_stable_content_id((1, 2))


def test_generate_stable_content_id_unhashable_type_raises_by_default() -> None:
    with pytest.raises(KeyError):
        generate_stable_content_id(object())


def test_generate_stable_content_id_ignore_unhashable_does_not_raise_warns() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        HEX_PATTERN.match(generate_stable_content_id(object(), ignore_unhashable=True))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_generate_stable_content_id_ignore_unhashable_does_not_raise() -> None:
    assert HEX_PATTERN.match(generate_stable_content_id(object(), ignore_unhashable=True))


def test_generate_stable_content_id_uses_given_registry() -> None:
    registry = HasherRegistry({object: StringHasher()})
    assert generate_stable_content_id("meow", registry=registry) == generate_stable_content_id(
        "meow", registry=registry
    )


def test_generate_stable_content_id_custom_registry_is_honored() -> None:
    custom_registry = HasherRegistry({object: ReprHasher()})
    assert generate_stable_content_id("meow") != generate_stable_content_id(
        "meow", registry=custom_registry
    )


def test_generate_stable_content_id_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        generate_stable_content_id({"a": 1}, 32)  # type: ignore[misc]  # ty: ignore[too-many-positional-arguments]


def test_generate_stable_content_id_golden_vector() -> None:
    assert generate_stable_content_id({"a": [1, 2]}, length=16) == "8884c93b49393739"


def test_generate_stable_content_id_golden_vector_default_length() -> None:
    assert generate_stable_content_id({"source": "cats.txt", "page": 1}) == (
        "f65b60565db14f8b43dc9aa7077c4f390b7a9a70a389b24d2b336d237981c06e"
    )


def test_generate_stable_content_id_golden_vector_maximum_length() -> None:
    assert generate_stable_content_id({"source": "cats.txt", "page": 1}, length=128) == (
        "f65b60565db14f8b43dc9aa7077c4f390b7a9a70a389b24d2b336d237981c06e"
        "5541d95f94e7fca1933c8d83ca7ab02c6121532e5092a156b8ea29651bf7d534"
    )


def test_generate_stable_content_id_differs_with_a_custom_registry() -> None:
    registry = HasherRegistry({object: ReprHasher()})
    assert generate_stable_content_id("meow", registry=registry) != generate_stable_content_id(
        "meow"
    )


###########################################
#     Tests for generate_stable_uuid5     #
###########################################


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
    assert generate_stable_uuid5(
        {"info": {"year": 2024, "topic": "cats"}}
    ) == generate_stable_uuid5({"info": {"topic": "cats", "year": 2024}})


def test_generate_stable_uuid5_list_input() -> None:
    assert UUID_PATTERN.match(generate_stable_uuid5([1, 2, 3]))


def test_generate_stable_uuid5_scalar_input() -> None:
    assert UUID_PATTERN.match(generate_stable_uuid5(42))


def test_generate_stable_uuid5_tuple_and_list_different_uuid() -> None:
    # The default registry tags each sequence family, so a list and an
    # equivalent tuple derive different UUIDs.
    assert generate_stable_uuid5([1, 2]) != generate_stable_uuid5((1, 2))


def test_generate_stable_uuid5_matches_uuid5_of_hash_object() -> None:
    # generate_stable_uuid5 is defined as uuid5 of hash_object's output
    # under a fixed namespace - this pins that relationship so the two
    # helpers cannot silently drift apart from each other.
    data = {"source": "cats.txt", "page": 1}
    assert generate_stable_uuid5(data) == str(uuid.uuid5(_NAMESPACE, hash_object(data, length=128)))


def test_generate_stable_uuid5_unhashable_type_raises_by_default() -> None:
    with pytest.raises(KeyError):
        generate_stable_uuid5(object())


def test_generate_stable_uuid5_ignore_unhashable_does_not_raise_warns() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        UUID_PATTERN.match(generate_stable_uuid5(object(), ignore_unhashable=True))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_generate_stable_uuid5_ignore_unhashable_does_not_raise() -> None:
    assert UUID_PATTERN.match(generate_stable_uuid5(object(), ignore_unhashable=True))


@pytest.mark.filterwarnings("ignore::UserWarning")
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


def test_generate_stable_uuid5_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        generate_stable_uuid5({"a": 1}, None)  # type: ignore[misc]  # ty: ignore[too-many-positional-arguments]


def test_generate_stable_uuid5_golden_vector() -> None:
    assert generate_stable_uuid5({"a": [1, 2]}) == "06ae7b88-52be-5262-ae72-b9b4174e193a"


def test_generate_stable_uuid5_golden_vector_nested_dict() -> None:
    assert (
        generate_stable_uuid5({"source": "cats.txt", "page": 1})
        == "3a4d214a-e7be-5842-a015-6be4af56a7ee"
    )


def test_generate_stable_uuid5_golden_vector_list() -> None:
    assert generate_stable_uuid5([1, 2, 3]) == "262fdcc3-2be3-5ce8-be86-ff0eec580ab6"


def test_generate_stable_uuid5_golden_vector_custom_namespace() -> None:
    assert (
        generate_stable_uuid5({"source": "cats.txt", "page": 1}, namespace=uuid.UUID(int=0))
        == "ffc04f29-021a-5e4d-8536-95ea21d9500f"
    )

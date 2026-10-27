from __future__ import annotations

import re
import uuid

import pytest

from coola.hashing import (
    HasherRegistry,
    ReprHasher,
    StringHasher,
    hash_object,
    interface as hashing_interface,
    register_hashers,
)
from coola.identifier.uuid5 import _NAMESPACE, generate_stable_uuid5


class MyInt(int):
    pass


UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$")


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


def test_generate_stable_uuid5_does_not_mutate_default_registry_when_custom_registry_given() -> (
    None
):
    custom_registry = HasherRegistry({object: StringHasher()})
    generate_stable_uuid5("meow", registry=custom_registry)
    assert not hashing_interface._default_registry.is_built()


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
        generate_stable_uuid5({"a": 1}, None)  # type: ignore[misc]


def test_generate_stable_uuid5_golden_vector() -> None:
    assert generate_stable_uuid5({"a": [1, 2]}) == "06ae7b88-52be-5262-ae72-b9b4174e193a"


def test_generate_stable_uuid5_ignores_hashers_registered_in_default_registry() -> None:
    expected = generate_stable_uuid5(MyInt(1))
    register_hashers({MyInt: StringHasher()})
    assert generate_stable_uuid5(MyInt(1)) == expected

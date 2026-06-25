from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from datetime import date, datetime, timezone

import pytest

from coola.hashing import (
    DefaultHasher,
    HasherRegistry,
    SequenceHasher,
    get_default_registry,
    hash_object,
    register_hashers,
)


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the singleton registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class CustomList(list):
    r"""Custom class that inherits from list."""


##################################
#     Tests for hash_object      #
##################################


def test_hash_object_with_int() -> None:
    assert hash_object(5) == "70b201352f24bf1c9770b99f8f71201821411cf414377c9b8c2dbcee61db87d6"


def test_hash_object_with_string() -> None:
    assert (
        hash_object("hello") == "324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf"
    )


def test_hash_object_with_list() -> None:
    assert (
        hash_object([1, 2, 3]) == "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de"
    )


def test_hash_object_with_dict() -> None:
    assert (
        hash_object({"a": 1, "b": 2})
        == "a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f"
    )


def test_hash_object_with_date() -> None:
    assert (
        hash_object(date(2021, 1, 1))
        == "f2b4c6a9941206bb6fc3b4b9c1104d8c05264985c009e2e1c7c840aaeda00dac"
    )


def test_hash_object_with_datetime() -> None:
    assert (
        hash_object(datetime(2021, 1, 1, tzinfo=timezone.utc))
        == "7b9000123bc9220f9759ddc4ae7e7780c16935c8ed6d9417b82c41500ebb3967"
    )


def test_hash_object_returns_str() -> None:
    assert isinstance(hash_object(42), str)


def test_hash_object_is_deterministic() -> None:
    assert hash_object([1, 2, 3]) == hash_object([1, 2, 3])


def test_hash_object_with_custom_registry() -> None:
    # A registry with only DefaultHasher treats lists as leaf values.
    registry = HasherRegistry({object: DefaultHasher()})
    assert hash_object([1, 2, 3], registry=registry) == hash_object(
        str([1, 2, 3]), registry=registry
    )


def test_hash_object_with_custom_length() -> None:
    result = hash_object("hello", length=16)
    assert isinstance(result, str)
    assert len(result) == 16


def test_hash_object_uses_default_registry_when_none() -> None:
    assert hash_object([1, 2, 3]) == hash_object([1, 2, 3], registry=get_default_registry())


#########################################
#     Tests for register_hashers        #
#########################################


def test_register_hashers_adds_to_default_registry() -> None:
    register_hashers({CustomList: SequenceHasher()})
    assert get_default_registry().has_hasher(CustomList)


def test_register_hashers_with_exist_ok_true() -> None:
    register_hashers({CustomList: DefaultHasher()})
    register_hashers({CustomList: SequenceHasher()}, exist_ok=True)


def test_register_hashers_with_exist_ok_false() -> None:
    register_hashers({CustomList: DefaultHasher()})
    with pytest.raises(RuntimeError, match=r"already registered"):
        register_hashers({CustomList: SequenceHasher()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    assert isinstance(get_default_registry(), HasherRegistry)


def test_get_default_registry_returns_singleton() -> None:
    assert get_default_registry() is get_default_registry()


def test_get_default_registry_singleton_persists_modifications() -> None:
    registry1 = get_default_registry()
    assert not registry1.has_hasher(CustomList)
    registry1.register(CustomList, SequenceHasher())
    assert registry1.has_hasher(CustomList)

    registry2 = get_default_registry()
    assert registry1 is registry2
    assert registry2.has_hasher(CustomList)


def test_get_default_registry_registers_object() -> None:
    assert get_default_registry().has_hasher(object)


def test_get_default_registry_registers_numeric_types() -> None:
    registry = get_default_registry()
    assert registry.has_hasher(bool)
    assert registry.has_hasher(int)
    assert registry.has_hasher(float)
    assert registry.has_hasher(complex)


def test_get_default_registry_registers_str() -> None:
    assert get_default_registry().has_hasher(str)


def test_get_default_registry_registers_datetime_types() -> None:
    registry = get_default_registry()
    assert registry.has_hasher(date)
    assert registry.has_hasher(datetime)


def test_get_default_registry_registers_sequences() -> None:
    registry = get_default_registry()
    assert registry.has_hasher(list)
    assert registry.has_hasher(tuple)
    assert registry.has_hasher(Sequence)


def test_get_default_registry_registers_mappings() -> None:
    registry = get_default_registry()
    assert registry.has_hasher(dict)
    assert registry.has_hasher(Mapping)


def test_get_default_registry_can_hash_list() -> None:
    assert get_default_registry().hash([1, 2, 3]) == (
        "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de"
    )


def test_get_default_registry_can_hash_dict() -> None:
    assert get_default_registry().hash({"a": 1, "b": 2}) == (
        "a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f"
    )

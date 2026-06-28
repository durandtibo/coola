from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from datetime import date, datetime, timezone

import pytest

from coola.hashing import (
    DatetimeHasher,
    HasherRegistry,
    MappingHasher,
    ReprHasher,
    SequenceHasher,
    StringHasher,
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


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            5,
            "70b201352f24bf1c9770b99f8f71201821411cf414377c9b8c2dbcee61db87d6",
            id="int",
        ),
        pytest.param(
            3.14,
            "ab14d2d4a60939ddc6de342ae33b8a8da25564787546cbdfd6cf29164863ed4a",
            id="float",
        ),
        pytest.param(
            complex(1, 2),
            "672e535e27fd6becac3687e261cc67a9708673e67eac19d588d574efadafc1e0",
            id="complex",
        ),
        pytest.param(
            True,
            "b37c53228410790b2e6d4ab6eb00deb4e1e9b47e2075100b120e0abd777d0020",
            id="bool",
        ),
        pytest.param(
            "hello",
            "324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de",
            id="list",
        ),
        pytest.param(
            (1, 2, 3),
            "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de",
            id="tuple",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            "a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f",
            id="dict",
        ),
        pytest.param(
            date(2021, 1, 1),
            "f2b4c6a9941206bb6fc3b4b9c1104d8c05264985c009e2e1c7c840aaeda00dac",
            id="date",
        ),
        pytest.param(
            datetime(2021, 1, 1, tzinfo=timezone.utc),
            "7b9000123bc9220f9759ddc4ae7e7780c16935c8ed6d9417b82c41500ebb3967",
            id="datetime",
        ),
    ],
)
def test_hash_object_parametrized(data: object, expected: str) -> None:
    assert hash_object(data) == expected


def test_hash_object_returns_str() -> None:
    assert isinstance(hash_object(42), str)


def test_hash_object_is_deterministic() -> None:
    assert hash_object([1, 2, 3]) == hash_object([1, 2, 3])


def test_hash_object_with_custom_registry() -> None:
    # A registry with only ReprHasher treats lists as leaf values.
    registry = HasherRegistry({object: ReprHasher()})
    assert hash_object([1, 2, 3], registry=registry) == hash_object(repr([1, 2, 3]))


def test_hash_object_with_custom_length() -> None:
    result = hash_object("hello", length=16)
    assert isinstance(result, str)
    assert len(result) == 16


def test_hash_object_uses_default_registry_when_none() -> None:
    assert hash_object([1, 2, 3]) == hash_object([1, 2, 3], registry=get_default_registry())


def test_hash_object_nested_structure() -> None:
    assert hash_object({"a": [1, 2], "b": [3, 4]}) == (
        "fe7eca5d3348be5060774aab9a95169595884dbb3d1fb7ddc318b1123eadc32b"
    )


#########################################
#     Tests for register_hashers        #
#########################################


def test_register_hashers_adds_to_default_registry() -> None:
    register_hashers({CustomList: SequenceHasher()})
    assert isinstance(get_default_registry().find_hasher(CustomList), SequenceHasher)


def test_register_hashers_with_exist_ok_true() -> None:
    register_hashers({CustomList: ReprHasher()})
    register_hashers({CustomList: SequenceHasher()}, exist_ok=True)


def test_register_hashers_with_exist_ok_false() -> None:
    register_hashers({CustomList: ReprHasher()})
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
    registry1.register(CustomList, SequenceHasher())
    assert isinstance(registry1.find_hasher(CustomList), SequenceHasher)

    registry2 = get_default_registry()
    assert registry1 is registry2
    assert isinstance(registry2.find_hasher(CustomList), SequenceHasher)


def test_get_default_registry_does_not_register_object() -> None:
    # object is no longer registered as a catch-all fallback.
    assert not get_default_registry().has_hasher(object)


def test_get_default_registry_registers_numeric_types() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(bool), ReprHasher)
    assert isinstance(registry.find_hasher(int), ReprHasher)
    assert isinstance(registry.find_hasher(float), ReprHasher)
    assert isinstance(registry.find_hasher(complex), ReprHasher)


def test_get_default_registry_registers_str() -> None:
    assert isinstance(get_default_registry().find_hasher(str), StringHasher)


def test_get_default_registry_registers_datetime_types() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(date), DatetimeHasher)
    assert isinstance(registry.find_hasher(datetime), DatetimeHasher)


def test_get_default_registry_registers_sequences() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(list), SequenceHasher)
    assert isinstance(registry.find_hasher(tuple), SequenceHasher)
    assert isinstance(registry.find_hasher(Sequence), SequenceHasher)


def test_get_default_registry_registers_mappings() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(dict), MappingHasher)
    assert isinstance(registry.find_hasher(Mapping), MappingHasher)


def test_get_default_registry_registers_none() -> None:
    assert isinstance(get_default_registry().find_hasher(type(None)), ReprHasher)


def test_get_default_registry_can_hash_list() -> None:
    assert get_default_registry().hash([1, 2, 3]) == (
        "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de"
    )


def test_get_default_registry_can_hash_dict() -> None:
    assert get_default_registry().hash({"a": 1, "b": 2}) == (
        "a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f"
    )

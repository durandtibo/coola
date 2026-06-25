from __future__ import annotations

from typing import Any

import pytest

from coola.hashing import (
    BaseHasher,
    DefaultHasher,
    HasherRegistry,
    MappingHasher,
    SequenceHasher,
    StringHasher,
)


class CustomList(list):
    r"""Custom class that inherits from list."""


#####################################
#     Tests for HasherRegistry      #
#####################################


def test_hasher_registry_init_empty() -> None:
    registry = HasherRegistry()
    assert len(registry._state) == 0


def test_hasher_registry_init_with_state() -> None:
    hasher = SequenceHasher()
    initial_state: dict[type, BaseHasher[Any]] = {list: hasher}
    registry = HasherRegistry(initial_state)

    assert list in registry._state
    assert registry._state[list] is hasher
    # Verify it's a copy — mutations to the source dict don't affect the registry.
    initial_state[tuple] = DefaultHasher()
    assert tuple not in registry._state


def test_hasher_registry_repr() -> None:
    assert repr(HasherRegistry()).startswith("HasherRegistry(")


def test_hasher_registry_str() -> None:
    assert str(HasherRegistry()).startswith("HasherRegistry(")


#####################################
#     Tests for register            #
#####################################


def test_hasher_registry_register_new_type() -> None:
    registry = HasherRegistry()
    hasher = SequenceHasher()
    registry.register(list, hasher)
    assert registry.has_hasher(list)
    assert registry._state[list] is hasher


def test_hasher_registry_register_existing_type_without_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    with pytest.raises(RuntimeError, match=r"already registered"):
        registry.register(list, MappingHasher(), exist_ok=False)


def test_hasher_registry_register_existing_type_with_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    hasher = MappingHasher()
    registry.register(list, hasher, exist_ok=True)
    assert registry._state[list] is hasher


#####################################
#     Tests for register_many       #
#####################################


def test_hasher_registry_register_many() -> None:
    registry = HasherRegistry()
    registry.register_many(
        {
            list: SequenceHasher(),
            dict: MappingHasher(),
            str: StringHasher(),
        }
    )
    assert registry.has_hasher(list)
    assert registry.has_hasher(dict)
    assert registry.has_hasher(str)


def test_hasher_registry_register_many_with_existing_type() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    with pytest.raises(RuntimeError, match=r"already registered"):
        registry.register_many({list: MappingHasher(), dict: MappingHasher()}, exist_ok=False)


def test_hasher_registry_register_many_with_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    hasher = MappingHasher()
    registry.register_many({list: hasher, dict: MappingHasher()}, exist_ok=True)
    assert registry._state[list] is hasher


#####################################
#     Tests for has_hasher          #
#####################################


def test_hasher_registry_has_hasher_true() -> None:
    assert HasherRegistry({list: SequenceHasher()}).has_hasher(list)


def test_hasher_registry_has_hasher_false() -> None:
    assert not HasherRegistry().has_hasher(list)


def test_hasher_registry_has_hasher_does_not_check_mro() -> None:
    # has_hasher checks direct registration only — not MRO fallback.
    assert not HasherRegistry({object: DefaultHasher()}).has_hasher(list)


#####################################
#     Tests for find_hasher         #
#####################################


def test_hasher_registry_find_hasher_direct_match() -> None:
    hasher = SequenceHasher()
    registry = HasherRegistry({list: hasher})
    assert registry.find_hasher(list) is hasher


def test_hasher_registry_find_hasher_mro_lookup() -> None:
    # CustomList inherits from list, so it resolves to the list hasher.
    hasher = SequenceHasher()
    registry = HasherRegistry({list: hasher})
    assert registry.find_hasher(CustomList) is hasher


def test_hasher_registry_find_hasher_most_specific() -> None:
    base_hasher = SequenceHasher()
    specific_hasher = MappingHasher()
    registry = HasherRegistry(
        {object: DefaultHasher(), list: base_hasher, CustomList: specific_hasher}
    )
    assert registry.find_hasher(CustomList) is specific_hasher


def test_hasher_registry_find_hasher_abc_not_in_mro() -> None:
    from collections.abc import Sequence

    # Sequence is an ABC not in list.__mro__, so list falls back to object.
    registry = HasherRegistry({object: DefaultHasher(), Sequence: SequenceHasher()})
    assert isinstance(registry.find_hasher(list), DefaultHasher)


#####################################
#     Tests for hash                #
#####################################


def test_hasher_registry_hash_with_list() -> None:
    assert (
        HasherRegistry({object: DefaultHasher(), list: SequenceHasher()}).hash([1, 2, 3])
        == "e30f3d309eab8b8216b15ef153005972ce61c8c64c55f78075630089aed023de"
    )


def test_hasher_registry_hash_with_dict() -> None:
    assert (
        HasherRegistry({object: DefaultHasher(), dict: MappingHasher()}).hash({"a": 1, "b": 2})
        == "a3ecbdde9e227bcdae038eb86746b0fccb90939d8e7eeac55513423219ffa02f"
    )


def test_hasher_registry_hash_with_nested_structure() -> None:
    registry = HasherRegistry(
        {
            object: DefaultHasher(),
            list: SequenceHasher(),
            dict: MappingHasher(),
        }
    )
    assert (
        registry.hash({"a": [1, 2], "b": [3, 4]})
        == "fe7eca5d3348be5060774aab9a95169595884dbb3d1fb7ddc318b1123eadc32b"
    )


def test_hasher_registry_hash_with_empty_list() -> None:
    assert (
        HasherRegistry({list: SequenceHasher()}).hash([])
        == "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"
    )


def test_hasher_registry_hash_with_empty_dict() -> None:
    assert (
        HasherRegistry({dict: MappingHasher()}).hash({})
        == "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"
    )


def test_hasher_registry_hash_with_length() -> None:
    result = HasherRegistry({object: DefaultHasher()}).hash(42, length=16)
    assert result == "57b43cf02666687a"
    assert len(result) == 16


def test_hasher_registry_hash_registry_isolation() -> None:
    hasher1 = SequenceHasher()
    hasher2 = MappingHasher()
    registry1 = HasherRegistry({list: hasher1})
    registry2 = HasherRegistry({list: hasher2})
    assert registry1._state[list] is hasher1
    assert registry2._state[list] is hasher2

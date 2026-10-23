from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from coola.hashing import (
    BaseHasher,
    HasherRegistry,
    MappingHasher,
    ReprHasher,
    SequenceHasher,
    StrHasher,
    StringHasher,
    get_default_registry,
)
from coola.registry import AlreadyRegisteredError, TypeNotRegisteredError
from tests.unit.hashing.helpers import Unhashable


class CustomList(list):
    r"""Custom class that inherits from list."""


####################################
#     Tests for HasherRegistry     #
####################################


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
    initial_state[tuple] = ReprHasher()
    assert tuple not in registry._state


def test_hasher_registry_repr() -> None:
    assert repr(HasherRegistry()).startswith("HasherRegistry(")


def test_hasher_registry_str() -> None:
    assert str(HasherRegistry()).startswith("HasherRegistry(")


##############################
#     Tests for register     #
##############################


def test_hasher_registry_register_new_type() -> None:
    registry = HasherRegistry()
    hasher = SequenceHasher()
    registry.register(list, hasher)
    assert registry.has_hasher(list)
    assert registry._state[list] is hasher


def test_hasher_registry_register_existing_type_without_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register(list, MappingHasher(), exist_ok=False)


def test_hasher_registry_register_existing_type_with_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    hasher = MappingHasher()
    registry.register(list, hasher, exist_ok=True)
    assert registry._state[list] is hasher


###################################
#     Tests for register_many     #
###################################


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
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register_many({list: MappingHasher(), dict: MappingHasher()}, exist_ok=False)


def test_hasher_registry_register_many_with_exist_ok() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    hasher = MappingHasher()
    registry.register_many({list: hasher, dict: MappingHasher()}, exist_ok=True)
    assert registry._state[list] is hasher


################################
#     Tests for has_hasher     #
################################


def test_hasher_registry_has_hasher_true() -> None:
    assert HasherRegistry({list: SequenceHasher()}).has_hasher(list)


def test_hasher_registry_has_hasher_false() -> None:
    assert not HasherRegistry().has_hasher(list)


def test_hasher_registry_has_hasher_does_not_check_mro() -> None:
    # has_hasher checks direct registration only — not MRO fallback.
    assert not HasherRegistry({object: ReprHasher()}).has_hasher(list)


#################################
#     Tests for find_hasher     #
#################################


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
        {object: ReprHasher(), list: base_hasher, CustomList: specific_hasher}
    )
    assert registry.find_hasher(CustomList) is specific_hasher


def test_hasher_registry_find_hasher_abc_not_in_mro() -> None:
    # Sequence is an ABC not in list.__mro__, so list falls back to object.
    registry = HasherRegistry({object: ReprHasher(), Sequence: SequenceHasher()})
    assert isinstance(registry.find_hasher(list), ReprHasher)


##########################
#     Tests for hash     #
##########################


def test_hasher_registry_hash_with_list() -> None:
    assert (
        HasherRegistry({object: StrHasher(), list: SequenceHasher()}).hash([1, 2, 3])
        == "bfe86f0fc5750a217446b5a88d2bfc543b33c63fa190489e917b16a0bc792cce"
    )


def test_hasher_registry_hash_with_dict() -> None:
    assert (
        HasherRegistry({object: StrHasher(), dict: MappingHasher()}).hash({"a": 1, "b": 2})
        == "0fde8be56160bfbace2c2f3ce1200a3a0b1355d40defca6fd85d8a423099a90a"
    )


def test_hasher_registry_hash_with_nested_structure() -> None:
    registry = HasherRegistry(
        {
            object: StrHasher(),
            list: SequenceHasher(),
            dict: MappingHasher(),
        }
    )
    assert (
        registry.hash({"a": [1, 2], "b": [3, 4]})
        == "6371c181bcdd8a939db72ab6345e8cb2ce13a1aa6ed806f0ccb4123b6149f84f"
    )


def test_hasher_registry_hash_with_empty_list() -> None:
    assert (
        HasherRegistry({list: SequenceHasher()}).hash([])
        == "844dea01562659240f7c438c4bc74c7713eb68c6e4d693a30bda1798aaa972d7"
    )


def test_hasher_registry_hash_with_empty_dict() -> None:
    assert (
        HasherRegistry({dict: MappingHasher()}).hash({})
        == "d0184328135d1a4b3a9492b8d5a214b49e15af380c2458d8a2671ecef2d2ee34"
    )


def test_hasher_registry_hash_with_length() -> None:
    result = HasherRegistry({object: ReprHasher()}).hash(42, length=16)
    assert result == "4d693658248572a8"
    assert len(result) == 16


def test_hasher_registry_hash_registry_isolation() -> None:
    hasher1 = SequenceHasher()
    hasher2 = MappingHasher()
    registry1 = HasherRegistry({list: hasher1})
    registry2 = HasherRegistry({list: hasher2})
    assert registry1._state[list] is hasher1
    assert registry2._state[list] is hasher2


def test_hasher_registry_hash_unhashable_raises_by_default() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        HasherRegistry().hash(Unhashable())


def test_hasher_registry_hash_unhashable_raises_when_explicitly_false() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        HasherRegistry().hash(Unhashable(), ignore_unhashable=False)


def test_hasher_registry_hash_ignore_unhashable_returns_str() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        result = HasherRegistry().hash(Unhashable(), ignore_unhashable=True)
    assert isinstance(result, str)


def test_hasher_registry_hash_ignore_unhashable_respects_length() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        result = HasherRegistry().hash(Unhashable(), length=16, ignore_unhashable=True)
    assert len(result) == 16


def test_hasher_registry_hash_ignore_unhashable_is_deterministic() -> None:
    registry = HasherRegistry()
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        first = registry.hash(Unhashable(), ignore_unhashable=True)
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        second = registry.hash(Unhashable(), ignore_unhashable=True)
    assert first == second


def test_hasher_registry_hash_ignore_unhashable_nested_in_list() -> None:
    # No `object` fallback registered, so the `Unhashable` element would
    # normally be unresolvable; ignore_unhashable=True must handle it.
    registry = HasherRegistry({int: StrHasher(), list: SequenceHasher()})
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        result = registry.hash([1, Unhashable(), 3], ignore_unhashable=True)
    assert isinstance(result, str)


def test_hasher_registry_hash_ignore_unhashable_nested_in_list_raises_by_default() -> None:
    # No `object` fallback registered, so an `Unhashable` element inside
    # the list cannot be resolved.
    registry = HasherRegistry({list: SequenceHasher()})
    with pytest.raises(KeyError, match=r"is not registered"):
        registry.hash([1, Unhashable(), 3])


def test_hasher_registry_hash_ignore_unhashable_nested_in_dict() -> None:
    # No `object` fallback registered, so the `Unhashable` value would
    # normally be unresolvable; ignore_unhashable=True must handle it.
    registry = HasherRegistry({str: StrHasher(), dict: MappingHasher()})
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        result = registry.hash({"a": Unhashable()}, ignore_unhashable=True)
    assert isinstance(result, str)


def test_hasher_registry_hash_ignore_unhashable_nested_in_dict_raises_by_default() -> None:
    registry = HasherRegistry({str: StrHasher(), dict: MappingHasher()})
    with pytest.raises(KeyError, match=r"is not registered"):
        registry.hash({"a": Unhashable()})


def test_hasher_registry_accepts_protocol_implementation_without_subclassing() -> None:
    class PlainHasher:
        def hash(
            self,
            data: Any,  # noqa: ARG002
            registry: HasherRegistry,  # noqa: ARG002
            *,
            length: int = 64,
            ignore_unhashable: bool = False,  # noqa: ARG002
        ) -> str:
            return "a" * length

    registry = HasherRegistry({object: PlainHasher()})
    assert registry.hash(1, length=4) == "aaaa"


def test_hasher_registry_hash_ignore_unhashable_warns() -> None:
    with pytest.warns(UserWarning, match="placeholder"):
        HasherRegistry().hash(Unhashable(), ignore_unhashable=True)


def test_hasher_registry_hash_self_referencing_raises() -> None:
    registry = HasherRegistry({list: SequenceHasher()})
    data: list = []
    data.append(data)
    with pytest.raises(ValueError, match="self-referencing"):
        registry.hash(data)


def test_hasher_registry_hash_same_object_twice_is_not_a_cycle() -> None:
    registry = HasherRegistry({list: SequenceHasher(), int: StrHasher()})
    shared = [1]
    assert isinstance(registry.hash([shared, shared]), str)


########################################
#     Tests for the fallbacks          #
########################################


class Marked:
    r"""A type matched by a fallback."""


def is_marked(data_type: type) -> bool:
    return issubclass(data_type, Marked)


def test_hasher_registry_find_hasher_uses_the_fallback() -> None:
    hasher = StrHasher()
    registry = HasherRegistry({list: SequenceHasher()})
    registry.register_fallback(is_marked, hasher)
    assert registry.find_hasher(Marked) is hasher


def test_hasher_registry_find_hasher_fallback_for_a_subclass() -> None:
    class Child(Marked): ...

    hasher = StrHasher()
    registry = HasherRegistry()
    registry.register_fallback(is_marked, hasher)
    assert registry.find_hasher(Child) is hasher


def test_hasher_registry_find_hasher_registered_type_wins_over_the_fallback() -> None:
    registered = ReprHasher()
    registry = HasherRegistry({Marked: registered})
    registry.register_fallback(is_marked, StrHasher())
    assert registry.find_hasher(Marked) is registered


def test_hasher_registry_find_hasher_registered_parent_wins_over_the_fallback() -> None:
    class Child(Marked): ...

    registered = ReprHasher()
    registry = HasherRegistry({Marked: registered})
    registry.register_fallback(is_marked, StrHasher())
    assert registry.find_hasher(Child) is registered


def test_hasher_registry_find_hasher_first_matching_fallback_wins() -> None:
    first, second = StrHasher(), ReprHasher()
    registry = HasherRegistry()
    registry.register_fallback(is_marked, first)
    registry.register_fallback(is_marked, second)
    assert registry.find_hasher(Marked) is first


def test_hasher_registry_find_hasher_skips_the_fallback_that_does_not_match() -> None:
    first, second = StrHasher(), ReprHasher()
    registry = HasherRegistry()
    registry.register_fallback(lambda _: False, first)
    registry.register_fallback(is_marked, second)
    assert registry.find_hasher(Marked) is second


def test_hasher_registry_find_hasher_no_matching_fallback_raises() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    with pytest.raises(KeyError, match="is not registered"):
        registry.find_hasher(Unhashable)


def test_hasher_registry_object_hasher_makes_the_fallbacks_unreachable() -> None:
    catch_all = ReprHasher()
    registry = HasherRegistry({object: catch_all})
    registry.register_fallback(is_marked, StrHasher())
    assert registry.find_hasher(Marked) is catch_all


class ConstantHasher(BaseHasher[Any]):
    def hash(
        self,
        data: Any,  # noqa: ARG002
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        return "c" * length


def test_hasher_registry_fallback_is_used_by_hash() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, ConstantHasher())
    assert registry.hash(Marked()) == "c" * 64


def test_hasher_registry_has_hasher_ignores_the_fallbacks() -> None:
    registry = HasherRegistry()
    registry.register_fallback(is_marked, StrHasher())
    assert not registry.has_hasher(Marked)


def test_hasher_registry_repr_shows_the_fallbacks() -> None:
    registry = HasherRegistry()
    assert "fallbacks" not in repr(registry)
    registry.register_fallback(is_marked, StrHasher())
    assert "fallbacks" in repr(registry)
    assert "StrHasher()" in repr(registry)


def test_hasher_registry_unhashable_error_names_the_remedies() -> None:
    registry = HasherRegistry()
    with pytest.raises(KeyError) as exc_info:
        registry.hash(Unhashable())
    message = str(exc_info.value)
    assert "register a hasher" in message
    assert "hash()" in message
    assert "ignore_unhashable=True" in message
    assert "Unhashable" in message


def test_hasher_registry_unhashable_error_is_a_type_not_registered_error() -> None:
    with pytest.raises(TypeNotRegisteredError):
        HasherRegistry().hash(Unhashable())


def test_hasher_registry_hash_reraises_a_key_error_that_is_not_a_missing_type() -> None:
    class BrokenRegistry(HasherRegistry):
        def find_hasher(self, data_type: type) -> Any:  # noqa: ARG002
            msg = "broken"
            raise KeyError(msg)

    with pytest.raises(KeyError, match="broken") as exc_info:
        BrokenRegistry().hash(1)
    assert not isinstance(exc_info.value, TypeNotRegisteredError)


def test_hasher_registry_hash_deeply_nested_data_raises_value_error() -> None:
    registry = get_default_registry()
    data: list = []
    for _ in range(5000):
        data = [data]
    with pytest.raises(ValueError, match="nested too deeply"):
        registry.hash(data)
    # The registry is still usable afterwards.
    assert registry.hash([1])

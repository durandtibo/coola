from __future__ import annotations

import abc
import contextlib
import copy
import pickle
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, is_dataclass
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock

import pytest

from coola.registry import (
    AlreadyRegisteredError,
    KeyNotRegisteredError,
    TypeNotRegisteredError,
    TypeRegistry,
)

if TYPE_CHECKING:
    import threading

##################################
#     Tests for TypeRegistry     #
##################################


def test_type_registry_init_empty() -> None:
    """Test creating an empty registry."""
    registry = TypeRegistry[str]()
    assert len(registry) == 0


def test_type_registry_init_with_initial_state() -> None:
    """Test creating a registry with initial state."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert len(registry) == 2


def test_type_registry_init_with_initial_state_get_first() -> None:
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert registry.get(int) == "integer"


def test_type_registry_init_with_initial_state_get_second() -> None:
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert registry.get(float) == "float"


def test_type_registry_init_with_initial_state_content() -> None:
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert registry._state == {int: "integer", float: "float"}


def test_type_registry_init_copies_initial_state() -> None:
    """Test that initial state is copied, not referenced."""
    initial: dict[type, str] = {int: "integer"}
    registry = TypeRegistry[str](initial)
    initial[float] = "float"
    assert len(registry) == 1


def test_type_registry_init_copies_initial_state_has() -> None:
    initial: dict[type, str] = {int: "integer"}
    registry = TypeRegistry[str](initial)
    initial[float] = "float"
    assert not registry.has(float)


def test_type_registry_init_copies_initial_state_content() -> None:
    initial: dict[type, str] = {int: "integer"}
    registry = TypeRegistry[str](initial)
    initial[float] = "float"
    assert registry._state == {int: "integer"}


def test_type_registry_repr() -> None:
    assert repr(TypeRegistry[str]()).startswith("TypeRegistry(")


def test_type_registry_str() -> None:
    assert str(TypeRegistry[str]()).startswith("TypeRegistry(")


def test_type_registry_clear_empty_registry() -> None:
    """Test clearing an empty registry."""
    registry = TypeRegistry[str]()
    registry.clear()
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_clear_empty_registry_resets_the_cache() -> None:
    """Test clearing an empty registry."""
    registry = TypeRegistry[str]()
    registry.clear()
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_clear_populated_registry() -> None:
    """Test clearing a populated registry."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    registry.clear()
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_clear_populated_registry_resets_the_cache() -> None:
    """Test clearing a populated registry."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    registry.clear()
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_equal_true() -> None:
    assert TypeRegistry[str]({int: "integer", float: "float"}).equal(
        TypeRegistry[str]({int: "integer", float: "float"})
    )


def test_type_registry_equal_true_same() -> None:
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert registry.equal(registry)


def test_type_registry_equal_false_different_state() -> None:
    assert not TypeRegistry[str]({int: "integer", float: "float"}).equal(TypeRegistry[str]())


def test_type_registry_equal_false_different_type() -> None:
    assert not TypeRegistry[str]().equal(42)


def test_type_registry_equal_false_different_type_child() -> None:
    class ChildRegistry(TypeRegistry[str]): ...

    assert not TypeRegistry[str]().equal(ChildRegistry())


def test_type_registry_get_existing_key() -> None:
    """Test getting an existing key."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert registry.get(int) == "integer"


def test_type_registry_get_missing_key_return_none() -> None:
    """Test that getting a missing key returns None."""
    registry = TypeRegistry[str]()
    assert registry.get(int) is None


def test_type_registry_get_missing_key_return_custom_default() -> None:
    """Test that getting a missing key returns the given default
    value."""
    registry = TypeRegistry[str]()
    assert registry.get(int, default="missing") == "missing"


def test_type_registry_get_existing_key_ignores_default() -> None:
    """Test that getting an existing key ignores the default value."""
    registry = TypeRegistry[str]({int: "integer"})
    assert registry.get(int, default="missing") == "integer"


def test_type_registry_has_existing_key() -> None:
    """Test checking existence of an existing key."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert registry.has(int)


def test_type_registry_has_missing_key() -> None:
    """Test checking existence of a missing key."""
    registry = TypeRegistry[str]()
    assert not registry.has(int)


def test_type_registry_register_basic() -> None:
    """Test basic registration of a key-value pair."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    assert registry.equal(TypeRegistry[str]({int: "integer"}))


def test_type_registry_register_basic_resets_the_cache() -> None:
    """Test basic registration of a key-value pair."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_register_duplicate_raises_error() -> None:
    """Test that registering a duplicate key raises RuntimeError."""
    registry = TypeRegistry[str]({int: "integer"})
    with pytest.raises(
        AlreadyRegisteredError, match=r"A value is already registered for <class 'int'>"
    ):
        registry.register(int, "int")


def test_type_registry_register_duplicate_with_exist_ok() -> None:
    """Test that exist_ok=True allows overwriting."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    registry.register(int, "int", exist_ok=True)
    assert registry.equal(TypeRegistry[str]({int: "int"}))


def test_type_registry_register_duplicate_with_exist_ok_resets_the_cache() -> None:
    """Test that exist_ok=True allows overwriting."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    registry.register(int, "int", exist_ok=True)
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_register_multiple_keys() -> None:
    """Test registering multiple different keys."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    registry.register(float, "float")
    assert registry.equal(TypeRegistry[str]({int: "integer", float: "float"}))


def test_type_registry_register_multiple_keys_resets_the_cache() -> None:
    """Test registering multiple different keys."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    registry.register(float, "float")
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_register_many_basic() -> None:
    """Test bulk registration with register_many."""
    registry = TypeRegistry[str]()
    registry.register_many({int: "integer", float: "float"})
    assert registry.equal(TypeRegistry[str]({int: "integer", float: "float"}))


def test_type_registry_register_many_basic_resets_the_cache() -> None:
    """Test bulk registration with register_many."""
    registry = TypeRegistry[str]()
    registry.register_many({int: "integer", float: "float"})
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_register_many_empty_mapping() -> None:
    """Test register_many with an empty mapping."""
    registry = TypeRegistry[str]()
    registry.register_many({})
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_register_many_empty_mapping_resets_the_cache() -> None:
    """Test register_many with an empty mapping."""
    registry = TypeRegistry[str]()
    registry.register_many({})
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_register_many_duplicate_raises_error() -> None:
    """Test that register_many raises error on duplicate without
    exist_ok."""
    registry = TypeRegistry[str]({int: "integer"})
    with pytest.raises(AlreadyRegisteredError, match=r"Types already registered"):
        registry.register_many({int: "integer", float: "float"})


def test_type_registry_register_many_with_exist_ok() -> None:
    """Test register_many with exist_ok=True allows overwriting."""
    registry = TypeRegistry[str]({int: "integer"})
    registry.register_many({int: "int", float: "float"}, exist_ok=True)
    assert registry.equal(TypeRegistry[str]({int: "int", float: "float"}))


def test_type_registry_register_many_with_exist_ok_resets_the_cache() -> None:
    """Test register_many with exist_ok=True allows overwriting."""
    registry = TypeRegistry[str]({int: "integer"})
    registry.register_many({int: "int", float: "float"}, exist_ok=True)
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_resolve_direct_match() -> None:
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert registry.resolve(int) == "integer"


def test_type_registry_resolve_direct_match_second_type() -> None:
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert registry.resolve(float) == "float"


def test_type_registry_resolve_via_mro() -> None:
    registry = TypeRegistry[str]({object: "object", int: "integer"})
    # bool inherits from int
    assert registry.resolve(bool) == "integer"


def test_type_registry_resolve_via_mro_object() -> None:
    registry = TypeRegistry[str]({object: "object", int: "integer"})
    # str inherits from object
    assert registry.resolve(str) == "object"


def test_type_registry_resolve_via_mro_populates_the_cache() -> None:
    registry = TypeRegistry[str]({object: "object", int: "integer"})
    registry.resolve(bool)
    registry.resolve(str)
    assert registry._cache == {bool: "integer", str: "object"}


class Animal: ...


class Dog(Animal): ...


class Poodle(Dog): ...


def test_type_registry_resolve_custom_class_hierarchy() -> None:
    registry = TypeRegistry[str]({Animal: "animal", Dog: "dog"})
    assert registry.resolve(Dog) == "dog"


def test_type_registry_resolve_custom_class_hierarchy_subclass() -> None:
    registry = TypeRegistry[str]({Animal: "animal", Dog: "dog"})
    assert registry.resolve(Poodle) == "dog"


def test_type_registry_resolve_custom_class_hierarchy_base() -> None:
    registry = TypeRegistry[str]({Animal: "animal", Dog: "dog"})
    assert registry.resolve(Animal) == "animal"


def test_type_registry_resolve_custom_class_hierarchy_cache() -> None:
    registry = TypeRegistry[str]({Animal: "animal", Dog: "dog"})
    registry.resolve(Dog)
    registry.resolve(Poodle)
    registry.resolve(Animal)
    assert registry._cache == {Animal: "animal", Poodle: "dog", Dog: "dog"}


def test_type_registry_resolve_missing_type_raises_keyerror() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError, match=r"Type \'<class \'int\'>\' is not registered"):
        registry.resolve(int)


def test_type_registry_not_registered_message_of_getitem_is_the_one_of_resolve() -> None:
    """``resolve`` and ``__getitem__`` must raise ``KeyError`` with
    identical wording for the same "not registered" condition."""
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError) as resolve_exc_info:
        registry.resolve(int)
    with pytest.raises(KeyError) as getitem_exc_info:
        _ = registry[int]
    assert str(resolve_exc_info.value) == str(getitem_exc_info.value)


def test_type_registry_not_registered_message_of_unregister_is_the_one_of_resolve() -> None:
    """``resolve`` and ``unregister`` must raise ``KeyError`` with
    identical wording for the same "not registered" condition."""
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError) as resolve_exc_info:
        registry.resolve(int)
    with pytest.raises(KeyError) as unregister_exc_info:
        registry.unregister(int)
    assert str(resolve_exc_info.value) == str(unregister_exc_info.value)


def test_type_registry_resolve_populates_the_cache() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(int)
    assert registry._cache == {int: "object"}


def test_type_registry_resolve_uses_cache() -> None:
    registry = TypeRegistry[str]({object: "object"})
    result1 = registry.resolve(int)
    assert registry.resolve(int) == result1


def _make_types(num_types: int) -> list[type]:
    return [type(f"Type{i}", (), {}) for i in range(num_types)]


def _resolve_all(registry: TypeRegistry[str], types: list[type]) -> None:
    for tp in types:
        registry.resolve(tp)


def test_type_registry_resolve_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the resolve() cache never grows past its maximum size."""
    monkeypatch.setattr("coola.registry.type._MAX_CACHE_SIZE", 8)
    registry = TypeRegistry[str]({object: "object"})
    _resolve_all(registry, _make_types(8))
    assert len(registry._cache) == 8


def test_type_registry_resolve_cache_is_reset_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the resolve() cache is reset when full."""
    monkeypatch.setattr("coola.registry.type._MAX_CACHE_SIZE", 8)
    registry = TypeRegistry[str]({object: "object"})
    types = _make_types(9)
    _resolve_all(registry, types[:8])
    registry.resolve(types[8])
    assert registry._cache == {types[8]: "object"}


def test_type_registry_resolve_cache_holds_every_resolved_type() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(int)
    registry.resolve(str)
    assert registry._cache == {int: "object", str: "object"}


def test_type_registry_resolve_miss_is_not_cached() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(TypeNotRegisteredError):
        registry.resolve(int)
    assert registry._cache == {}


def test_type_registry_resolve_unhashable_key_raises_type_error() -> None:
    registry = TypeRegistry[str]({object: "object"})
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.resolve([])  # ty: ignore[invalid-argument-type]


def test_type_registry_resolve_registered_after_cached_lookup_is_seen() -> None:
    registry = TypeRegistry[str]({object: "object", int: "int"})
    registry.resolve(bool)
    registry.register(bool, "bool")
    assert registry.resolve(bool) == "bool"


def test_type_registry_resolve_cache_cleared_on_change() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(int)
    registry.register(int, "integer")
    assert registry._cache == {}


def test_type_registry_resolve_after_change_uses_the_new_state() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(int)
    registry.register(int, "integer")
    assert registry.resolve(int) == "integer"


def test_type_registry_resolve_most_specific_type() -> None:
    registry = TypeRegistry[str]({object: "object", int: "int", float: "float"})
    # Should get most specific match
    assert registry.resolve(bool) == "int"


def test_type_registry_resolve_most_specific_type_exact_match() -> None:
    registry = TypeRegistry[str]({object: "object", int: "int", float: "float"})
    assert registry.resolve(int) == "int"


def test_type_registry_unregister_existing_key() -> None:
    """Test unregistering an existing key."""
    registry = TypeRegistry[str]({int: "integer"})
    assert registry.unregister(int) == "integer"


def test_type_registry_unregister_existing_key_removes_the_key() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    registry.unregister(int)
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_unregister_existing_key_resets_the_cache() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    registry.unregister(int)
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_unregister_missing_key_raises_error() -> None:
    """Test that unregistering a missing key raises KeyError."""
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError, match=r"Type \'<class \'int\'>\' is not registered"):
        registry.unregister(int)


def test_type_registry_unregister_reduces_length() -> None:
    """Test that unregister reduces registry length."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    registry.unregister(int)
    assert len(registry) == 1


def test_type_registry_unregister_keeps_the_other_keys() -> None:
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    registry.unregister(int)
    assert registry.equal(TypeRegistry[str]({float: "float"}))


def test_type_registry_unregister_one_of_two_keys_resets_the_cache() -> None:
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    registry.unregister(int)
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_items() -> None:
    """Test items() method returns key-value pairs."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert list(registry.items()) == [(int, "integer"), (float, "float")]


def test_type_registry_items_returns_copy() -> None:
    """Test that items() returns a copy, not a view of internal
    state."""
    registry = TypeRegistry[str]({int: "integer"})
    items = iter(registry.items())
    registry[float] = "float"
    assert list(items) == [(int, "integer")]


def test_type_registry_items_empty() -> None:
    """Test items() on empty registry."""
    registry = TypeRegistry[str]()
    assert list(registry.items()) == []


def test_type_registry_keys() -> None:
    """Test keys() method returns all keys."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert list(registry.keys()) == [int, float]


def test_type_registry_keys_returns_copy() -> None:
    """Test that keys() returns a copy, not a view of internal state."""
    registry = TypeRegistry[str]({int: "integer"})
    keys = list(registry.keys())
    registry[float] = "float"
    assert len(keys) == 1


def test_type_registry_keys_empty() -> None:
    """Test keys() on empty registry."""
    registry = TypeRegistry[str]()
    assert list(registry.keys()) == []


def test_type_registry_values() -> None:
    """Test values() method returns all values."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert list(registry.values()) == ["integer", "float"]


def test_type_registry_values_returns_copy() -> None:
    """Test that values() returns a copy, not a view of internal
    state."""
    registry = TypeRegistry[str]({int: "integer"})
    values = list(registry.values())
    registry[float] = "float"
    assert len(values) == 1


def test_type_registry_values_empty() -> None:
    """Test values() on empty registry."""
    registry = TypeRegistry[str]()
    assert list(registry.values()) == []


# Test operator overloading


def test_type_registry_contains_operator() -> None:
    """Test the 'in' operator."""
    registry = TypeRegistry[str]({int: "integer"})
    assert int in registry


def test_type_registry_not_contains_operator() -> None:
    """Test the 'not in' operator."""
    registry = TypeRegistry[str]({int: "integer"})
    assert float not in registry


def test_type_registry_getitem_operator() -> None:
    """Test the bracket access operator."""
    registry = TypeRegistry[str]({int: "integer"})
    assert registry[int] == "integer"


def test_type_registry_getitem_missing_raises_error() -> None:
    """Test that bracket access on missing key raises KeyError."""
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError, match=r"Type \'<class \'int\'>\' is not registered"):
        _ = registry[int]


def test_type_registry_setitem_operator() -> None:
    """Test the bracket assignment operator."""
    registry = TypeRegistry[str]()
    registry[int] = "integer"
    assert registry.equal(TypeRegistry[str]({int: "integer"}))


def test_type_registry_setitem_operator_resets_the_cache() -> None:
    """Test the bracket assignment operator."""
    registry = TypeRegistry[str]()
    registry[int] = "integer"
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_setitem_overwrites() -> None:
    """Test that bracket assignment overwrites existing values."""
    registry = TypeRegistry[str]()
    registry[int] = "int"
    registry[int] = "integer"
    assert registry.equal(TypeRegistry[str]({int: "integer"}))


def test_type_registry_setitem_overwrites_resets_the_cache() -> None:
    """Test that bracket assignment overwrites existing values."""
    registry = TypeRegistry[str]()
    registry[int] = "int"
    registry[int] = "integer"
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_delitem_operator() -> None:
    """Test the del operator."""
    registry = TypeRegistry[str]({int: "integer"})
    del registry[int]
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_delitem_operator_resets_the_cache() -> None:
    """Test the del operator."""
    registry = TypeRegistry[str]({int: "integer"})
    del registry[int]
    assert registry._cache == {}  # verify the cache is reset because the state changed


def test_type_registry_delitem_missing_raises_error() -> None:
    """Test that del on missing key raises KeyError."""
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError):
        del registry[int]


def test_type_registry_iter() -> None:
    """Test iterating over registry yields keys."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert list(registry) == [int, float]


def test_type_registry_iter_returns_copy() -> None:
    """Test that iteration uses a snapshot of keys."""
    registry = TypeRegistry[str]({int: "integer"})
    iterator = iter(registry)
    registry[float] = "float"
    keys = list(iterator)
    assert keys == [int]


def test_type_registry_iter_empty() -> None:
    """Test iterating over an empty registry."""
    registry = TypeRegistry[str]()
    assert list(registry) == []


def test_type_registry_len_operator() -> None:
    """Test the len() function."""
    assert len(TypeRegistry[str]()) == 0


def test_type_registry_len_operator_one_item() -> None:
    registry = TypeRegistry[str]()
    registry[int] = "integer"
    assert len(registry) == 1


def test_type_registry_len_operator_two_items() -> None:
    registry = TypeRegistry[str]()
    registry[int] = "integer"
    registry[float] = "float"
    assert len(registry) == 2


# Test with different types


def test_type_registry_registry_with_int_keys() -> None:
    """Test registry with integer values."""
    registry = TypeRegistry[int]()
    registry.register(int, 1)
    registry.register(float, 2)
    assert registry.get(int) == 1


def test_type_registry_registry_with_int_keys_second_value() -> None:
    registry = TypeRegistry[int]()
    registry.register(int, 1)
    registry.register(float, 2)
    assert registry.get(float) == 2


def test_type_registry_registry_with_int_keys_equal() -> None:
    registry = TypeRegistry[int]()
    registry.register(int, 1)
    registry.register(float, 2)
    assert registry.equal(TypeRegistry[int]({int: 1, float: 2}))


# Test resolve() with ABC virtual subclasses


class MyABC(abc.ABC):  # noqa: B024
    pass


class NotARealSubclass:
    pass


MyABC.register(NotARealSubclass)


def test_type_registry_virtual_subclass_is_seen_by_isinstance() -> None:
    """Test that isinstance() sees the virtual subclass relationship."""
    assert isinstance(NotARealSubclass(), MyABC)


def test_type_registry_resolve_does_not_match_abc_virtual_subclass() -> None:
    """Test that resolve() does not follow ABC virtual subclass
    registration, since it only walks __mro__ (static inheritance)."""
    registry = TypeRegistry[str]()
    registry.register(MyABC, "abc value")
    with pytest.raises(KeyError, match="is not registered"):
        registry.resolve(NotARealSubclass)


def test_type_registry_resolve_abc_virtual_subclass_registered_after_lookup() -> None:
    """Test that registering a value for an ABC after a virtual subclass
    has already been resolved does not retroactively make resolve()
    match the virtual subclass (the cache is not the cause: even an
    uncached lookup does not follow virtual subclass relationships)."""
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError, match="is not registered"):
        registry.resolve(NotARealSubclass)


def test_type_registry_resolve_abc_virtual_subclass_still_not_resolved_after_register() -> None:
    """Test that virtual subclasses never appear in __mro__, cache or no
    cache."""
    registry = TypeRegistry[str]()
    with contextlib.suppress(KeyError):
        registry.resolve(NotARealSubclass)
    # Register the ABC only after the failed lookup above.
    registry.register(MyABC, "abc value")
    with pytest.raises(KeyError, match="is not registered"):
        registry.resolve(NotARealSubclass)


def test_type_registry_resolve_registered_concrete_subclass_of_abc() -> None:
    """Test that registering the concrete subclass itself does work."""
    registry = TypeRegistry[str]({MyABC: "abc value"})
    registry.register(NotARealSubclass, "concrete value")
    assert registry.resolve(NotARealSubclass) == "concrete value"


def test_type_registry_resolve_raises_type_not_registered_error() -> None:
    registry = TypeRegistry[str]()
    registry.register(str, "string")
    with pytest.raises(TypeNotRegisteredError, match=r"MRO: int, object"):
        registry.resolve(int)


def test_type_registry_resolve_raises_a_key_error() -> None:
    registry = TypeRegistry[str]({str: "string"})
    with pytest.raises(TypeNotRegisteredError) as exc_info:
        registry.resolve(int)
    assert isinstance(exc_info.value, KeyError)


def test_type_registry_resolve_error_lists_the_registered_types() -> None:
    registry = TypeRegistry[str]({str: "string"})
    with pytest.raises(TypeNotRegisteredError) as exc_info:
        registry.resolve(int)
    assert "Registered types: str" in str(exc_info.value)


def test_type_registry_not_registered_msg_without_mro() -> None:
    """The MRO section is omitted when the key has no ``__mro__``."""
    registry = TypeRegistry[str]({int: "integer"})
    assert "MRO" not in registry._not_registered_msg(cast("type", "not a type"))


def test_type_registry_not_registered_msg_without_mro_starts_with_the_key() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    msg = registry._not_registered_msg(cast("type", "not a type"))
    assert msg.startswith("Type 'not a type' is not registered")


def test_type_registry_not_registered_msg_without_mro_lists_the_registered_types() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    msg = registry._not_registered_msg(cast("type", "not a type"))
    assert "Registered types: int" in msg


def test_type_registry_not_registered_msg_empty_registry() -> None:
    """The MRO section is kept for an empty registry."""
    msg = TypeRegistry[str]()._not_registered_msg(int)
    assert "MRO: int, object" in msg


def test_type_registry_not_registered_msg_empty_registry_omits_registered_types() -> None:
    """The registered-types section is omitted for an empty registry."""
    msg = TypeRegistry[str]()._not_registered_msg(int)
    assert "Registered types" not in msg


def test_type_registry_resolve_returns_cached_value_set_during_lock_acquisition() -> None:
    # Covers the double-checked-locking branch: if another caller populated the
    # cache while this one waited for the lock, the value is returned without
    # resolving again.
    registry = TypeRegistry[str]()
    registry.register(int, "int")

    class PopulatingLock:
        def __enter__(self) -> None:
            registry._cache = {int: "cached-by-another-caller"}

        def __exit__(self, *args: object) -> None:
            return None

    registry._lock = cast("threading.RLock", PopulatingLock())
    assert registry.resolve(int) == "cached-by-another-caller"


###############################################
#     Tests for the validation of the keys    #
###############################################


def test_type_registry_register_non_type_key_raises() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.register("abc", "value")  # ty: ignore[invalid-argument-type]


def test_type_registry_register_non_type_key_raises_error() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.register("abc", "value")  # ty: ignore[invalid-argument-type]


def test_type_registry_register_non_type_key_does_not_change_state() -> None:
    registry = TypeRegistry[str]()
    with contextlib.suppress(TypeError):
        registry.register("abc", "value")  # ty: ignore[invalid-argument-type]
    assert len(registry) == 0


def test_type_registry_register_many_non_type_key_raises() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.register_many({int: "integer", "abc": "value"})  # ty: ignore[invalid-argument-type]


def test_type_registry_register_many_non_type_key_raises_error() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.register_many({int: "integer", "abc": "value"})  # ty: ignore[invalid-argument-type]


def test_type_registry_register_many_non_type_key_does_not_change_state() -> None:
    registry = TypeRegistry[str]()
    with contextlib.suppress(TypeError):
        registry.register_many({int: "integer", "abc": "value"})  # ty: ignore[invalid-argument-type]
    assert len(registry) == 0


def test_type_registry_init_non_type_key_raises() -> None:
    with pytest.raises(TypeError, match="Expected a type as key"):
        TypeRegistry[str]({"abc": "value"})  # ty: ignore[invalid-argument-type]


def test_type_registry_setitem_non_type_key_raises() -> None:
    registry = TypeRegistry[str]()
    key: Any = "abc"
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry[key] = "value"


def test_type_registry_not_registered_msg_same_qualname_is_qualified() -> None:
    def make_class() -> type:
        class Foo:
            pass

        return Foo

    first, second = make_class(), make_class()
    msg = TypeRegistry[str]({first: "first"})._not_registered_msg(second)
    assert "Registered types: tests.unit.registry.test_type." in msg


def test_type_registry_many_already_registered_msg_is_sorted() -> None:
    registry = TypeRegistry[str]({str: "a", int: "b", float: "c"})
    with pytest.raises(AlreadyRegisteredError, match=r"float.*int.*str"):
        registry.register_many({str: "x", int: "y", float: "z"})


###########################################
#     Tests for the lazy registration     #
###########################################


def test_type_registry_register_lazy_resolve_loads_the_group_of_the_type() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.resolve(Decimal) == "decimal"


def test_type_registry_register_lazy_resolve_loads_the_group_of_a_base_class() -> None:
    class MyDecimal(Decimal):
        pass

    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.resolve(MyDecimal) == "decimal"


def test_type_registry_register_lazy_resolve_does_not_load_the_group_of_another_type() -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    registry = TypeRegistry[str]({object: "object"})
    registry.register_lazy("decimal", factory)
    registry.resolve(int)
    factory.assert_not_called()


def test_type_registry_register_lazy_resolve_uses_the_loaded_value_over_the_parent() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.resolve(Decimal) == "decimal"


def test_type_registry_register_lazy_has_loads_the_group_of_the_type() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.has(Decimal)


def test_type_registry_register_lazy_has_does_not_load_the_group_of_another_type() -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    registry.has(int)
    factory.assert_not_called()


def test_type_registry_register_lazy_register_loads_the_group_of_the_type() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        registry.register(Decimal, "custom")


def test_type_registry_register_lazy_register_exist_ok_replaces_the_loaded_value() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry.register(Decimal, "custom", exist_ok=True)
    assert registry.resolve(Decimal) == "custom"


def test_type_registry_register_lazy_does_not_replace_a_registered_type() -> None:
    registry = TypeRegistry[str]({Decimal: "custom"})
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.resolve(Decimal) == "custom"


def test_type_registry_register_lazy_len_loads_all_the_groups() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry.register_lazy("fractions", lambda: {Fraction: "fraction"})
    assert len(registry) == 2


def test_type_registry_register_lazy_resolve_clears_the_cache_of_the_loaded_group() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry.resolve(int)
    assert registry.resolve(Decimal) == "decimal"


def test_type_registry_register_lazy_validates_the_loaded_keys() -> None:
    registry = TypeRegistry[str]()

    def load() -> Any:
        return {"Decimal": "decimal"}

    registry.register_lazy("decimal", load)
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.resolve(Decimal)


def test_type_registry_resolve_no_match_raises() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        TypeRegistry[str]().resolve(int)


def test_type_registry_register_lazy_contains_non_type_is_false() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    key: Any = "abc"
    assert key not in registry


def test_type_registry_register_lazy_has_non_type_is_false() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    key: Any = None
    assert not registry.has(key)


def test_type_registry_register_lazy_get_non_type_returns_default() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    key: Any = "abc"
    assert registry.get(key, "default") == "default"


def test_type_registry_register_lazy_getitem_non_type_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    key: Any = "abc"
    with pytest.raises(KeyError):
        registry[key]


def test_type_registry_register_lazy_resolve_non_type_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    key: Any = "abc"
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.resolve(key)


def test_type_registry_register_lazy_resolve_non_type_does_not_load_the_groups() -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    key: Any = "abc"
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.resolve(key)
    factory.assert_not_called()


def test_type_registry_register_lazy_unregister_non_type_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    key: Any = "abc"
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.unregister(key)


#############################################
#     Tests for the fallbacks of resolve    #
#############################################


@dataclass
class Point:
    x: int = 1


class Base:
    pass


class Child(Base):
    pass


def is_child(dtype: type) -> bool:
    return issubclass(dtype, Child)


def is_base(dtype: type) -> bool:
    return issubclass(dtype, Base)


def is_sequence(dtype: type) -> bool:
    return issubclass(dtype, Sequence)


def raises(_dtype: type) -> bool:
    msg = "predicate failed"
    raise RuntimeError(msg)


def test_type_registry_fallbacks_empty() -> None:
    assert list(TypeRegistry[str]().fallbacks()) == []


def test_type_registry_register_fallback_resolve_dataclass() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_fallback(is_dataclass, "dataclass")
    assert registry.resolve(Point) == "dataclass"


def test_type_registry_register_fallback_does_not_match_other_types() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_fallback(is_dataclass, "dataclass")
    assert registry.resolve(int) == "object"


def test_type_registry_register_fallback_matches_virtual_subclass() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_fallback(is_sequence, "sequence")
    assert registry.resolve(deque) == "sequence"


def test_type_registry_register_fallback_without_object_raises_if_no_match() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.resolve(int)


def test_type_registry_register_fallback_does_not_add_an_entry() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    assert len(registry) == 0


def test_type_registry_register_fallback_is_seen_after_cached_lookup() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(Point)
    registry.register_fallback(is_dataclass, "dataclass")
    assert registry.resolve(Point) == "dataclass"


def test_type_registry_register_fallback_resolve_object_checks_the_fallbacks() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_fallback(lambda dtype: dtype is object, "fallback")
    assert registry.resolve(object) == "fallback"


@pytest.mark.parametrize(
    ("has_type", "has_fallback", "has_object", "expected"),
    [
        (True, True, True, "type"),
        (True, True, False, "type"),
        (True, False, True, "type"),
        (True, False, False, "type"),
        (False, True, True, "fallback"),
        (False, True, False, "fallback"),
        (False, False, True, "object"),
    ],
)
def test_type_registry_resolve_precedence(
    has_type: bool, has_fallback: bool, has_object: bool, expected: str
) -> None:
    registry = TypeRegistry[str]()
    if has_type:
        registry.register(Base, "type")
    if has_fallback:
        registry.register_fallback(is_base, "fallback")
    if has_object:
        registry.register(object, "object")
    assert registry.resolve(Child) == expected


def test_type_registry_resolve_precedence_none_raises() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        TypeRegistry[str]().resolve(Child)


def test_type_registry_resolve_fallbacks_are_tried_in_registration_order() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_base, "first")
    registry.register_fallback(is_child, "second")
    assert registry.resolve(Child) == "first"


def test_type_registry_resolve_fallback_exist_ok_keeps_the_position() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_base, "first")
    registry.register_fallback(is_child, "second")
    registry.register_fallback(is_base, "replaced", exist_ok=True)
    assert [value for _, value in registry.fallbacks()] == ["replaced", "second"]


def test_type_registry_resolve_handler_shared_by_object_and_base_class() -> None:
    """Regression test: the handler of a base class is used for a
    dataclass that inherits from it, even if the same handler is
    registered for ``object``."""

    @dataclass
    class Derived(Base):
        x: int = 1

    shared = "shared"
    registry = TypeRegistry[str]({object: shared, Base: shared})
    registry.register_fallback(is_dataclass, "dataclass")
    assert registry.resolve(Derived) == "shared"


def test_type_registry_resolve_fallback_is_cached() -> None:
    calls = []

    def predicate(dtype: type) -> bool:
        calls.append(dtype)
        return True

    registry = TypeRegistry[str]()
    registry.register_fallback(predicate, "fallback")
    registry.resolve(Child)
    registry.resolve(Child)
    assert calls == [Child]


def test_type_registry_resolve_fallback_that_raises_propagates() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_fallback(raises, "fallback")
    with pytest.raises(RuntimeError, match="predicate failed"):
        registry.resolve(int)


def test_type_registry_resolve_fallback_that_raises_is_not_cached() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_fallback(raises, "fallback")
    with contextlib.suppress(RuntimeError):
        registry.resolve(int)
    assert registry._cache == {}


def test_type_registry_register_fallback_duplicate_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    with pytest.raises(AlreadyRegisteredError, match="A fallback is already registered"):
        registry.register_fallback(is_dataclass, "other")


def test_type_registry_register_fallback_duplicate_does_not_change_the_value() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    with contextlib.suppress(AlreadyRegisteredError):
        registry.register_fallback(is_dataclass, "other")
    assert list(registry.fallbacks()) == [(is_dataclass, "dataclass")]


def test_type_registry_register_fallback_exist_ok_replaces_the_value() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    registry.register_fallback(is_dataclass, "other", exist_ok=True)
    assert list(registry.fallbacks()) == [(is_dataclass, "other")]


def test_type_registry_register_fallback_exist_ok_is_keyword_only() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(TypeError, match="positional"):
        registry.register_fallback(is_dataclass, "dataclass", True)  # ty: ignore[too-many-positional-arguments]


def test_type_registry_unregister_fallback_returns_the_value() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    assert registry.unregister_fallback(is_dataclass) == "dataclass"


def test_type_registry_unregister_fallback_is_seen_after_cached_lookup() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_fallback(is_dataclass, "dataclass")
    registry.resolve(Point)
    registry.unregister_fallback(is_dataclass)
    assert registry.resolve(Point) == "object"


def test_type_registry_unregister_fallback_unknown_raises() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(KeyNotRegisteredError, match="No fallback is registered"):
        registry.unregister_fallback(is_dataclass)


def test_type_registry_unregister_fallback_unknown_is_a_key_error() -> None:
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError):
        registry.unregister_fallback(is_dataclass)


def test_type_registry_fallbacks_is_a_snapshot() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_base, "first")
    view = registry.fallbacks()
    registry.register_fallback(is_child, "second")
    assert list(view) == [(is_base, "first")]


def test_type_registry_fallbacks_follow_the_registration_order() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_child, "child")
    registry.register_fallback(is_base, "base")
    assert list(registry.fallbacks()) == [(is_child, "child"), (is_base, "base")]


def test_type_registry_copy_has_the_fallbacks() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    assert list(registry.copy().fallbacks()) == [(is_dataclass, "dataclass")]


def test_type_registry_copy_fallbacks_are_independent() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    other = registry.copy()
    other.register_fallback(is_base, "base")
    assert list(registry.fallbacks()) == [(is_dataclass, "dataclass")]


def test_type_registry_copy_unregister_fallback_does_not_change_the_source() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    registry.copy().unregister_fallback(is_dataclass)
    assert list(registry.fallbacks()) == [(is_dataclass, "dataclass")]


def test_type_registry_copy_resolves_with_the_fallbacks() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.register_fallback(is_dataclass, "dataclass")
    assert registry.copy().resolve(Point) == "dataclass"


def test_type_registry_clear_removes_the_fallbacks() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    registry.clear()
    assert list(registry.fallbacks()) == []


def test_type_registry_clear_is_seen_after_cached_lookup() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    registry.resolve(Point)
    registry.clear()
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.resolve(Point)


def test_type_registry_repr_with_fallbacks() -> None:
    registry = TypeRegistry[str]({int: "int"})
    registry.register_fallback(is_base, "base")
    assert repr(registry) == (
        f"TypeRegistry(\n  (<class 'int'>): int\n  (fallback) {is_base!r}: base\n)"
    )


def test_type_registry_str_with_fallbacks_and_pending_group() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_base, "base")
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert str(registry) == (
        f"TypeRegistry(\n  (fallback) {is_base!r}: base\n  <not loaded: decimal>\n)"
    )


def test_type_registry_equal_same_fallbacks() -> None:
    registry1 = TypeRegistry[str]()
    registry2 = TypeRegistry[str]()
    registry1.register_fallback(is_dataclass, "dataclass")
    registry2.register_fallback(is_dataclass, "dataclass")
    assert registry1.equal(registry2)


def test_type_registry_equal_different_fallbacks() -> None:
    registry1 = TypeRegistry[str]()
    registry2 = TypeRegistry[str]()
    registry1.register_fallback(is_dataclass, "dataclass")
    assert not registry1.equal(registry2)


def test_type_registry_equal_different_fallback_value() -> None:
    registry1 = TypeRegistry[str]()
    registry2 = TypeRegistry[str]()
    registry1.register_fallback(is_dataclass, "dataclass")
    registry2.register_fallback(is_dataclass, "other")
    assert not registry1.equal(registry2)


def test_type_registry_equal_other_type_is_false() -> None:
    assert not TypeRegistry[str]().equal(42)


def test_type_registry_equal_different_order_of_fallbacks() -> None:
    registry1 = TypeRegistry[str]()
    registry1.register_fallback(is_base, "base")
    registry1.register_fallback(is_child, "child")
    registry2 = TypeRegistry[str]()
    registry2.register_fallback(is_child, "child")
    registry2.register_fallback(is_base, "base")
    assert not registry1.equal(registry2)


#############################################
#     Tests for the lazy groups (groups)    #
#############################################


def lazy_decimal_entries() -> dict[type, str]:
    return {Decimal: "decimal"}


def test_type_registry_register_lazy_does_not_call_the_factory() -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    TypeRegistry[str]().register_lazy("decimal", factory)
    factory.assert_not_called()


def test_type_registry_register_lazy_calls_the_factory_once() -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    len(registry)
    len(registry)
    factory.assert_called_once_with()


def test_type_registry_register_lazy_packages_load_the_group_once() -> None:
    factory = Mock(return_value={Decimal: "decimal", Fraction: "fraction"})
    registry = TypeRegistry[str]()
    registry.register_lazy(("decimal", "fractions"), factory)
    registry.resolve(Decimal)
    registry.resolve(Fraction)
    factory.assert_called_once_with()


def test_type_registry_register_lazy_imports_the_factory_when_it_is_a_path() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", "tests.unit.registry.test_type:lazy_decimal_entries")
    assert registry.resolve(Decimal) == "decimal"


@pytest.mark.parametrize(
    "read",
    [
        pytest.param(len, id="len"),
        pytest.param(list, id="iter"),
        pytest.param(lambda r: list(r.items()), id="items"),
        pytest.param(lambda r: list(r.keys()), id="keys"),
        pytest.param(lambda r: list(r.values()), id="values"),
    ],
)
def test_type_registry_register_lazy_whole_content_loads_every_group(read: Any) -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry.register_lazy("fractions", lambda: {Fraction: "fraction"})
    read(registry)
    assert registry._state == {Decimal: "decimal", Fraction: "fraction"}


@pytest.mark.parametrize("show", [repr, str])
def test_type_registry_register_lazy_display_does_not_load(show: Any) -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    show(registry)
    factory.assert_not_called()


def test_type_registry_register_lazy_copy_does_not_load() -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    registry.copy()
    factory.assert_not_called()


def test_type_registry_register_lazy_repr_lists_the_pending_packages() -> None:
    registry = TypeRegistry[str]({int: "int"})
    registry.register_lazy(["fractions", "decimal"], lambda: {Decimal: "decimal"})
    assert repr(registry) == (
        "TypeRegistry(\n  (<class 'int'>): int\n  <not loaded: decimal, fractions>\n)"
    )


def test_type_registry_register_lazy_repr_after_loading_has_no_pending_line() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    len(registry)
    assert repr(registry) == "TypeRegistry(\n  (<class 'decimal.Decimal'>): decimal\n)"


def test_type_registry_register_lazy_copy_has_the_pending_groups() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.copy().resolve(Decimal) == "decimal"


def test_type_registry_register_lazy_copy_loads_the_groups_separately() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry.copy().resolve(Decimal)
    assert len(registry._state) == 0


def test_type_registry_register_lazy_clear_removes_the_groups() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry.clear()
    assert len(registry) == 0


def test_type_registry_register_lazy_clear_keeps_the_groups_of_the_original_copy() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry.copy().clear()
    assert len(registry) == 1


def test_type_registry_register_lazy_failing_factory_keeps_the_group() -> None:
    factory = Mock(side_effect=[ImportError("failed"), {Decimal: "decimal"}])
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    with pytest.raises(ImportError, match="failed"):
        registry.resolve(Decimal)
    assert registry.resolve(Decimal) == "decimal"


def test_type_registry_register_lazy_failing_factory_raises_again() -> None:
    factory = Mock(side_effect=ImportError("failed"))
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    with contextlib.suppress(ImportError), pytest.raises(ImportError, match="failed"):
        len(registry)
    with pytest.raises(ImportError, match="failed"):
        len(registry)


def test_type_registry_register_lazy_factory_can_register_in_the_registry() -> None:
    registry = TypeRegistry[str]()

    def factory() -> dict[type, str]:
        registry.register(int, "int")
        return {Decimal: "decimal"}

    registry.register_lazy("decimal", factory)
    assert registry.resolve(Decimal) == "decimal"


def test_type_registry_register_lazy_factory_registration_is_kept() -> None:
    registry = TypeRegistry[str]()

    def factory() -> dict[type, str]:
        registry.register(int, "int")
        return {Decimal: "decimal"}

    registry.register_lazy("decimal", factory)
    registry.resolve(Decimal)
    assert registry[int] == "int"


def test_type_registry_register_lazy_two_groups_with_the_same_package_are_both_loaded() -> None:
    """Regression test (F4): the second group used to replace the
    first."""
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry.register_lazy("decimal", lambda: {Fraction: "fraction"})
    len(registry)
    assert registry._state == {Decimal: "decimal", Fraction: "fraction"}


def test_type_registry_register_lazy_group_of_two_packages_then_one_of_them() -> None:
    """Regression test (F4): the alias that was reused used to leave a
    dangling name and raise a ``KeyError`` at the next load."""
    registry = TypeRegistry[str]()
    registry.register_lazy(["builtins", "decimal"], lambda: {Decimal: "decimal"})
    registry.register_lazy("builtins", lambda: {Fraction: "fraction"})
    len(registry)
    assert registry._state == {Decimal: "decimal", Fraction: "fraction"}


def test_type_registry_register_lazy_register_over_a_lazy_default_needs_exist_ok() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        registry.register(Decimal, "mine")


def test_type_registry_register_lazy_equal_copy_does_not_load() -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    registry.copy().equal(registry)
    factory.assert_not_called()


def test_type_registry_register_lazy_equal_loads_the_different_groups() -> None:
    registry1 = TypeRegistry[str]()
    registry1.register_lazy("decimal", lambda: {Decimal: "decimal"})
    registry2 = TypeRegistry[str]({Decimal: "decimal"})
    assert registry1.equal(registry2)


def test_type_registry_register_lazy_invalid_key_from_the_factory_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {"abc": "x"})  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="Expected a type as key"):
        len(registry)


def test_type_registry_register_lazy_group_loaded_by_another_factory_is_not_loaded_twice() -> None:
    registry = TypeRegistry[str]()
    second = Mock(return_value={Fraction: "fraction"})

    def first() -> dict[type, str]:
        registry.resolve(Fraction)  # loads the second group
        return {Decimal: "decimal"}

    registry.register_lazy("decimal", first)
    registry.register_lazy("fractions", second)
    len(registry)
    second.assert_called_once_with()


########################################
#     Tests for pickle and deepcopy    #
########################################


def round_trip(registry: TypeRegistry[str]) -> TypeRegistry[str]:
    return pickle.loads(pickle.dumps(registry))  # noqa: S301


def test_type_registry_pickle_keeps_the_entries() -> None:
    registry = TypeRegistry[str]({object: "object", int: "int"})
    assert round_trip(registry).equal(registry)


def test_type_registry_pickle_keeps_the_fallbacks() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(is_dataclass, "dataclass")
    assert list(round_trip(registry).fallbacks()) == [(is_dataclass, "dataclass")]


def test_type_registry_pickle_keeps_the_pending_groups() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", "tests.unit.registry.test_type:lazy_decimal_entries")
    assert round_trip(registry).resolve(Decimal) == "decimal"


def test_type_registry_pickle_does_not_load_the_pending_groups() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", "tests.unit.registry.test_type:lazy_decimal_entries")
    assert len(round_trip(registry)._state) == 0


def test_type_registry_pickle_resolves_after_the_round_trip() -> None:
    registry = TypeRegistry[str]({object: "object", int: "int"})
    assert round_trip(registry).resolve(bool) == "int"


def test_type_registry_pickle_drops_the_cache() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(int)
    assert round_trip(registry)._cache == {}


def test_type_registry_pickle_source_keeps_its_cache() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(int)
    pickle.dumps(registry)
    assert registry._cache == {int: "object"}


@pytest.mark.parametrize(
    "dtype",
    [
        MappingProxyType,
        type({}.keys()),
        type({}.values()),
        type({}.items()),
    ],
)
def test_type_registry_pickle_keeps_the_types_without_an_importable_name(dtype: type) -> None:
    registry = TypeRegistry[str]({dtype: "handler"})
    assert round_trip(registry)[dtype] == "handler"


def test_type_registry_pickle_keeps_the_order_of_the_entries() -> None:
    registry = TypeRegistry[str]({MappingProxyType: "a", int: "b", type({}.values()): "c"})
    assert list(round_trip(registry).keys()) == [MappingProxyType, int, type({}.values())]


def test_type_registry_deepcopy_keeps_the_entries() -> None:
    registry = TypeRegistry[str]({MappingProxyType: "a", int: "b"})
    assert copy.deepcopy(registry).equal(registry)


def test_type_registry_deepcopy_has_its_own_lock() -> None:
    registry = TypeRegistry[str]()
    assert copy.deepcopy(registry)._lock is not registry._lock


def test_type_registry_pickle_unpicklable_predicate_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_fallback(lambda dtype: dtype is int, "int")
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(registry)

from __future__ import annotations

import abc
import contextlib
import gc
import weakref
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import pytest

from coola.registry import AlreadyRegisteredError, TypeNotRegisteredError, TypeRegistry

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


def test_type_registry_resolve_cache_snapshot_not_mutated() -> None:
    """Test resolve() publishes a new snapshot instead of mutating the
    previous one (lock-free readers rely on this)."""
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(int)
    snapshot = registry._cache
    registry.resolve(str)
    assert snapshot == {int: "object"}


def test_type_registry_resolve_cache_snapshot_is_published() -> None:
    registry = TypeRegistry[str]({object: "object"})
    registry.resolve(int)
    registry.resolve(str)
    assert registry._cache == {int: "object", str: "object"}


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
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry["abc"] = "value"  # ty: ignore[invalid-key]


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


def test_type_registry_cache_does_not_keep_a_dynamic_class_alive() -> None:
    registry = TypeRegistry[str]({object: "obj"})
    dynamic = type("Dynamic", (), {})
    registry.resolve(dynamic)
    reference = weakref.ref(dynamic)
    del dynamic
    gc.collect()
    assert reference() is None


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
    registry.register_lazy("decimal", lambda: {"Decimal": "decimal"})
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.resolve(Decimal)


###################################
#     Tests for resolve_or        #
###################################


def test_type_registry_resolve_or_direct_match() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    assert registry.resolve_or(int) == "integer"


def test_type_registry_resolve_or_parent_match() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    assert registry.resolve_or(bool) == "integer"


def test_type_registry_resolve_or_no_match_returns_none() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    assert registry.resolve_or(str) is None


def test_type_registry_resolve_or_no_match_returns_the_default() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    assert registry.resolve_or(str, "default") == "default"


def test_type_registry_resolve_or_empty_registry() -> None:
    assert TypeRegistry[str]().resolve_or(int, "default") == "default"


def test_type_registry_resolve_or_does_not_build_the_error_message() -> None:
    registry = TypeRegistry[str]()
    with patch.object(TypeRegistry, "_not_registered_msg") as msg:
        registry.resolve_or(int)
    msg.assert_not_called()


def test_type_registry_resolve_or_caches_a_match() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    registry.resolve_or(bool)
    assert registry._cache == {bool: "integer"}


def test_type_registry_resolve_or_does_not_cache_a_miss() -> None:
    registry = TypeRegistry[str]({int: "integer"})
    registry.resolve_or(str)
    assert registry._cache == {}


def test_type_registry_resolve_or_registration_after_a_miss_is_used() -> None:
    registry = TypeRegistry[str]()
    registry.resolve_or(int)
    registry.register(int, "integer")
    assert registry.resolve_or(int) == "integer"


def test_type_registry_resolve_or_loads_the_group_of_the_type() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.resolve_or(Decimal) == "decimal"


def test_type_registry_resolve_no_match_raises() -> None:
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        TypeRegistry[str]().resolve(int)


def test_type_registry_register_lazy_contains_non_type_is_false() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert "abc" not in registry


def test_type_registry_register_lazy_has_non_type_is_false() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert not registry.has(None)


def test_type_registry_register_lazy_get_non_type_returns_default() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.get("abc", "default") == "default"


def test_type_registry_register_lazy_getitem_non_type_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    with pytest.raises(KeyError):
        registry["abc"]


def test_type_registry_register_lazy_resolve_non_type_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.resolve("abc")


def test_type_registry_register_lazy_resolve_non_type_does_not_load_the_groups() -> None:
    factory = Mock(return_value={Decimal: "decimal"})
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", factory)
    with pytest.raises(TypeError, match="Expected a type as key"):
        registry.resolve("abc")
    factory.assert_not_called()


def test_type_registry_register_lazy_unregister_non_type_raises() -> None:
    registry = TypeRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        registry.unregister("abc")


def test_type_registry_resolve_or_non_type_raises() -> None:
    with pytest.raises(TypeError, match="Expected a type as key"):
        TypeRegistry[str]().resolve_or("abc")

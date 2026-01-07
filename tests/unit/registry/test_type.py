from __future__ import annotations

import pytest

from coola.registry import TypeRegistry

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
    assert registry.get(int) == "integer"
    assert registry.get(float) == "float"
    assert registry._state == {int: "integer", float: "float"}


def test_type_registry_init_copies_initial_state() -> None:
    """Test that initial state is copied, not referenced."""
    initial: dict[type, str] = {int: "integer"}
    registry = TypeRegistry[str](initial)
    initial[float] = "float"
    assert len(registry) == 1
    assert not registry.has(float)
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


def test_type_registry_clear_populated_registry() -> None:
    """Test clearing a populated registry."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    registry.clear()
    assert registry.equal(TypeRegistry[str]())


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


def test_type_registry_register_duplicate_raises_error() -> None:
    """Test that registering a duplicate key raises RuntimeError."""
    registry = TypeRegistry[str]({int: "integer"})
    with pytest.raises(RuntimeError, match="A value is already registered for <class 'int'>"):
        registry.register(int, "int")


def test_type_registry_register_duplicate_with_exist_ok() -> None:
    """Test that exist_ok=True allows overwriting."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    registry.register(int, "int", exist_ok=True)
    assert registry.equal(TypeRegistry[str]({int: "int"}))


def test_type_registry_register_multiple_keys() -> None:
    """Test registering multiple different keys."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    registry.register(float, "float")
    assert registry.equal(TypeRegistry[str]({int: "integer", float: "float"}))


def test_type_registry_register_many_basic() -> None:
    """Test bulk registration with register_many."""
    registry = TypeRegistry[str]()
    registry.register_many({int: "integer", float: "float"})
    assert registry.equal(TypeRegistry[str]({int: "integer", float: "float"}))


def test_type_registry_register_many_empty_mapping() -> None:
    """Test register_many with an empty mapping."""
    registry = TypeRegistry[str]()
    registry.register_many({})
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_register_many_duplicate_raises_error() -> None:
    """Test that register_many raises error on duplicate without
    exist_ok."""
    registry = TypeRegistry[str]({int: "integer"})
    with pytest.raises(RuntimeError, match="Types already registered"):
        registry.register_many({int: "integer", float: "float"})


def test_type_registry_register_many_with_exist_ok() -> None:
    """Test register_many with exist_ok=True allows overwriting."""
    registry = TypeRegistry[str]({int: "integer"})
    registry.register_many({int: "int", float: "float"}, exist_ok=True)
    assert registry.equal(TypeRegistry[str]({int: "int", float: "float"}))


def test_type_registry_unregister_existing_key() -> None:
    """Test unregistering an existing key."""
    registry = TypeRegistry[str]({int: "integer"})
    value = registry.unregister(int)
    assert value == "integer"
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_unregister_missing_key_raises_error() -> None:
    """Test that unregistering a missing key raises KeyError."""
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError, match="Type <class 'int'> is not registered"):
        registry.unregister(int)


def test_type_registry_unregister_reduces_length() -> None:
    """Test that unregister reduces registry length."""
    registry = TypeRegistry[str]({int: "integer", float: "float"})
    assert len(registry) == 2
    registry.unregister(int)
    assert len(registry) == 1
    assert registry.equal(TypeRegistry[str]({float: "float"}))


# Test operator overloading


def test_type_registry_contains_operator() -> None:
    """Test the 'in' operator."""
    registry = TypeRegistry[str]({int: "integer"})
    assert int in registry
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


def test_type_registry_setitem_overwrites() -> None:
    """Test that bracket assignment overwrites existing values."""
    registry = TypeRegistry[str]()
    registry[int] = "int"
    registry[int] = "integer"
    assert registry.equal(TypeRegistry[str]({int: "integer"}))


def test_type_registry_delitem_operator() -> None:
    """Test the del operator."""
    registry = TypeRegistry[str]({int: "integer"})
    del registry[int]
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_delitem_missing_raises_error() -> None:
    """Test that del on missing key raises KeyError."""
    registry = TypeRegistry[str]()
    with pytest.raises(KeyError):
        del registry[int]


def test_type_registry_len_operator() -> None:
    """Test the len() function."""
    registry = TypeRegistry[str]()
    assert len(registry) == 0
    registry[int] = "integer"
    assert len(registry) == 1
    registry[float] = "float"
    assert len(registry) == 2


# Test with different types


def test_type_registry_registry_with_int_keys() -> None:
    """Test registry with integer keys."""
    registry = TypeRegistry[int]()
    registry.register(int, 1)
    registry.register(float, 2)
    assert registry.get(int) == 1
    assert registry.get(float) == 2
    assert registry.equal(TypeRegistry[int]({int: 1, float: 2}))

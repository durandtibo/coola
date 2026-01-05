from __future__ import annotations

import pytest

from coola.registry import Registry

##############################
#     Tests for Registry     #
##############################


def test_registry_init_empty() -> None:
    """Test creating an empty registry."""
    registry = Registry[str, int]()
    assert len(registry) == 0


def test_registry_init_with_initial_state() -> None:
    """Test creating a registry with initial state."""
    registry = Registry[str, int](initial_state={"a": 1, "b": 2, "c": 3})
    assert len(registry) == 3
    assert registry.get("a") == 1
    assert registry.get("b") == 2
    assert registry.get("c") == 3
    assert registry._registry == {"a": 1, "b": 2, "c": 3}


def test_registry_init_copies_initial_state() -> None:
    """Test that initial state is copied, not referenced."""
    initial = {"a": 1}
    registry = Registry[str, int](initial_state=initial)
    initial["b"] = 2
    assert len(registry) == 1
    assert not registry.has("b")
    assert registry._registry == {"a": 1}


def test_registry_repr() -> None:
    assert repr(Registry[str, int]()).startswith("Registry(")


def test_registry_str() -> None:
    assert str(Registry[str, int]()).startswith("Registry(")


def test_registry_clear_empty_registry() -> None:
    """Test clearing an empty registry."""
    registry = Registry[str, int]()
    registry.clear()
    assert registry.equal(Registry[str, int]())


def test_registry_clear_populated_registry() -> None:
    """Test clearing a populated registry."""
    registry = Registry[str, int]()
    registry.register_many({"a": 1, "b": 2, "c": 3})
    assert registry.equal(Registry[str, int]({"a": 1, "b": 2, "c": 3}))
    registry.clear()
    assert registry.equal(Registry[str, int]())


def test_registry_equal_true() -> None:
    assert Registry[str, int]({"key1": 42}).equal(Registry[str, int]({"key1": 42}))


def test_registry_equal_true_same() -> None:
    registry = Registry[str, int]({"key1": 42})
    assert registry.equal(registry)


def test_registry_equal_false_different_state() -> None:
    assert not Registry[str, int]({"key1": 42}).equal(Registry[str, int]())


def test_registry_equal_false_different_type() -> None:
    assert not Registry[str, int]({"key1": 42}).equal(42)


def test_registry_equal_false_different_type_child() -> None:
    class ChildRegistry(Registry[str, int]): ...

    assert not Registry[str, int]().equal(ChildRegistry())


def test_registry_get_existing_key() -> None:
    """Test getting an existing key."""
    registry = Registry[str, int]({"key1": 42})
    assert registry.get("key1") == 42


def test_registry_get_missing_key_raises_error() -> None:
    """Test that getting a missing key raises KeyError."""
    registry = Registry[str, int]()
    with pytest.raises(KeyError, match="Key 'missing' is not registered"):
        registry.get("missing")


def test_registry_has_existing_key() -> None:
    """Test checking existence of an existing key."""
    registry = Registry[str, int]({"key1": 42})
    assert registry.has("key1")


def test_registry_has_missing_key() -> None:
    """Test checking existence of a missing key."""
    registry = Registry[str, int]()
    assert not registry.has("missing")


def test_registry_register_basic() -> None:
    """Test basic registration of a key-value pair."""
    registry = Registry[str, int]()
    registry.register("key1", 42)
    assert registry.equal(Registry[str, int]({"key1": 42}))


def test_registry_register_duplicate_raises_error() -> None:
    """Test that registering a duplicate key raises RuntimeError."""
    registry = Registry[str, int]({"key1": 42})
    with pytest.raises(RuntimeError, match="A value is already registered for 'key1'"):
        registry.register("key1", 100)


def test_registry_register_duplicate_with_exist_ok() -> None:
    """Test that exist_ok=True allows overwriting."""
    registry = Registry[str, int]()
    registry.register("key1", 42)
    registry.register("key1", 100, exist_ok=True)
    assert registry.get("key1") == 100


def test_registry_register_multiple_keys() -> None:
    """Test registering multiple different keys."""
    registry = Registry[str, int]()
    registry.register("a", 1)
    registry.register("b", 2)
    registry.register("c", 3)
    assert registry.equal(Registry[str, int]({"a": 1, "b": 2, "c": 3}))


def test_registry_register_many_basic() -> None:
    """Test bulk registration with register_many."""
    registry = Registry[str, int]()
    registry.register_many({"a": 1, "b": 2, "c": 3})
    assert registry.equal(Registry[str, int]({"a": 1, "b": 2, "c": 3}))


def test_registry_register_many_empty_mapping() -> None:
    """Test register_many with an empty mapping."""
    registry = Registry[str, int]()
    registry.register_many({})
    assert registry.equal(Registry[str, int]())


def test_registry_register_many_duplicate_raises_error() -> None:
    """Test that register_many raises error on duplicate without
    exist_ok."""
    registry = Registry[str, int]()
    registry.register("a", 1)
    with pytest.raises(RuntimeError, match="Keys already registered"):
        registry.register_many({"a": 10, "b": 2})


def test_registry_register_many_with_exist_ok() -> None:
    """Test register_many with exist_ok=True allows overwriting."""
    registry = Registry[str, int]()
    registry.register("a", 1)
    registry.register_many({"a": 10, "b": 2}, exist_ok=True)
    assert registry.equal(Registry[str, int]({"a": 10, "b": 2}))


def test_registry_unregister_existing_key() -> None:
    """Test unregistering an existing key."""
    registry = Registry[str, int]()
    registry.register("key1", 42)
    value = registry.unregister("key1")
    assert value == 42
    assert registry.equal(Registry[str, int]())


def test_registry_unregister_missing_key_raises_error() -> None:
    """Test that unregistering a missing key raises KeyError."""
    registry = Registry[str, int]()
    with pytest.raises(KeyError, match="Key 'missing' is not registered"):
        registry.unregister("missing")


def test_registry_unregister_reduces_length() -> None:
    """Test that unregister reduces registry length."""
    registry = Registry[str, int]()
    registry.register("a", 1)
    registry.register("b", 2)
    assert len(registry) == 2
    registry.unregister("a")
    assert len(registry) == 1


# Test operator overloading


def test_registry_contains_operator() -> None:
    """Test the 'in' operator."""
    registry = Registry[str, int]({"key1": 42})
    assert "key1" in registry
    assert "missing" not in registry


def test_registry_getitem_operator() -> None:
    """Test the bracket access operator."""
    registry = Registry[str, int]({"key1": 42})
    assert registry["key1"] == 42


def test_registry_getitem_missing_raises_error() -> None:
    """Test that bracket access on missing key raises KeyError."""
    registry = Registry[str, int]()
    with pytest.raises(KeyError):
        _ = registry["missing"]


def test_registry_setitem_operator() -> None:
    """Test the bracket assignment operator."""
    registry = Registry[str, int]()
    registry["key1"] = 42
    assert registry.equal(Registry[str, int]({"key1": 42}))


def test_registry_setitem_overwrites() -> None:
    """Test that bracket assignment overwrites existing values."""
    registry = Registry[str, int]()
    registry["key1"] = 42
    registry["key1"] = 100
    assert registry.equal(Registry[str, int]({"key1": 100}))


def test_registry_delitem_operator() -> None:
    """Test the del operator."""
    registry = Registry[str, int]({"key1": 42})
    del registry["key1"]
    assert registry.equal(Registry[str, int]())


def test_registry_delitem_missing_raises_error() -> None:
    """Test that del on missing key raises KeyError."""
    registry = Registry[str, int]()
    with pytest.raises(KeyError):
        del registry["missing"]


def test_registry_len_operator() -> None:
    """Test the len() function."""
    registry = Registry[str, int]()
    assert len(registry) == 0
    registry.register("a", 1)
    assert len(registry) == 1
    registry.register("b", 2)
    assert len(registry) == 2


# Test with different types


def test_registry_registry_with_int_keys() -> None:
    """Test registry with integer keys."""
    registry = Registry[int, str]()
    registry.register(1, "one")
    registry.register(2, "two")
    assert registry.get(1) == "one"
    assert registry.get(2) == "two"


def test_registry_registry_with_tuple_keys() -> None:
    """Test registry with tuple keys."""
    registry = Registry[tuple, str]()
    registry.register((1, 2), "pair")
    assert registry.get((1, 2)) == "pair"


def test_registry_registry_with_complex_values() -> None:
    """Test registry with complex value types."""
    registry = Registry[str, list]()
    registry.register("list1", [1, 2, 3])
    registry.register("list2", ["a", "b", "c"])
    assert registry.get("list1") == [1, 2, 3]
    assert registry.get("list2") == ["a", "b", "c"]

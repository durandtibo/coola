from __future__ import annotations

import contextlib
import copy
import pickle

import pytest

from coola.registry import AlreadyRegisteredError, Registry, TypeRegistry
from coola.utils import fork
from tests.helpers import pickle_round_trip

##############################
#     Tests for Registry     #
##############################


def test_registry_init_empty() -> None:
    """Test creating an empty registry."""
    registry = Registry[str, int]()
    assert len(registry) == 0


def test_registry_init_with_initial_state() -> None:
    """Test creating a registry with initial state."""
    registry = Registry[str, int]({"a": 1, "b": 2, "c": 3})
    assert len(registry) == 3


@pytest.mark.parametrize(("key", "value"), [("a", 1), ("b", 2), ("c", 3)])
def test_registry_init_with_initial_state_get(key: str, value: int) -> None:
    registry = Registry[str, int]({"a": 1, "b": 2, "c": 3})
    assert registry.get(key) == value


def test_registry_init_with_initial_state_content() -> None:
    registry = Registry[str, int]({"a": 1, "b": 2, "c": 3})
    assert registry._state == {"a": 1, "b": 2, "c": 3}


def test_registry_init_copies_initial_state() -> None:
    """Test that initial state is copied, not referenced."""
    initial = {"a": 1}
    registry = Registry[str, int](initial)
    initial["b"] = 2
    assert len(registry) == 1


def test_registry_init_copies_initial_state_has() -> None:
    initial = {"a": 1}
    registry = Registry[str, int](initial)
    initial["b"] = 2
    assert not registry.has("b")


def test_registry_init_copies_initial_state_content() -> None:
    initial = {"a": 1}
    registry = Registry[str, int](initial)
    initial["b"] = 2
    assert registry._state == {"a": 1}


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
    registry = Registry[str, int]({"a": 1, "b": 2, "c": 3})
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


def test_registry_get_missing_key_return_none() -> None:
    """Test that getting a missing key returns None."""
    registry = Registry[str, int]()
    assert registry.get("missing") is None


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
    with pytest.raises(AlreadyRegisteredError, match=r"A value is already registered for 'key1'"):
        registry.register("key1", 100)


def test_registry_register_duplicate_with_exist_ok() -> None:
    """Test that exist_ok=True allows overwriting."""
    registry = Registry[str, int]()
    registry.register("key1", 42)
    registry.register("key1", 100, exist_ok=True)
    assert registry.equal(Registry[str, int]({"key1": 100}))


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
    registry = Registry[str, int]({"a": 1})
    with pytest.raises(AlreadyRegisteredError, match=r"Keys already registered"):
        registry.register_many({"a": 10, "b": 2})


def test_registry_register_many_with_exist_ok() -> None:
    """Test register_many with exist_ok=True allows overwriting."""
    registry = Registry[str, int]({"a": 1})
    registry.register_many({"a": 10, "b": 2}, exist_ok=True)
    assert registry.equal(Registry[str, int]({"a": 10, "b": 2}))


def test_registry_unregister_existing_key() -> None:
    """Test unregistering an existing key."""
    registry = Registry[str, int]()
    registry.register("key1", 42)
    assert registry.unregister("key1") == 42


def test_registry_unregister_existing_key_removes_the_key() -> None:
    registry = Registry[str, int]()
    registry.register("key1", 42)
    registry.unregister("key1")
    assert registry.equal(Registry[str, int]())


def test_registry_unregister_missing_key_raises_error() -> None:
    """Test that unregistering a missing key raises KeyError."""
    registry = Registry[str, int]()
    with pytest.raises(KeyError, match=r"Key 'missing' is not registered"):
        registry.unregister("missing")


def test_registry_unregister_reduces_length() -> None:
    """Test that unregister reduces registry length."""
    registry = Registry[str, int]({"a": 1, "b": 2})
    registry.unregister("a")
    assert len(registry) == 1


def test_registry_unregister_keeps_the_other_keys() -> None:
    registry = Registry[str, int]({"a": 1, "b": 2})
    registry.unregister("a")
    assert registry.equal(Registry[str, int]({"b": 2}))


def test_registry_items() -> None:
    """Test items() method returns key-value pairs."""
    registry = Registry[str, int]({"a": 1, "b": 2})
    assert list(registry.items()) == [("a", 1), ("b", 2)]


def test_registry_items_returns_copy() -> None:
    """Test that items() returns a copy, not a view of internal
    state."""
    registry = Registry[str, int]({"a": 1})
    items = iter(registry.items())
    registry["b"] = 2
    assert list(items) == [("a", 1)]


def test_registry_items_empty() -> None:
    """Test items() on empty registry."""
    registry = Registry[str, int]()
    assert list(registry.items()) == []


def test_registry_keys() -> None:
    """Test keys() method returns all keys."""
    registry = Registry[str, int]({"a": 1, "b": 2, "c": 3})
    assert list(registry.keys()) == ["a", "b", "c"]


def test_registry_keys_returns_copy() -> None:
    """Test that keys() returns a copy, not a view of internal state."""
    registry = Registry[str, int]({"a": 1})
    keys = list(registry.keys())
    registry["b"] = 2
    assert len(keys) == 1


def test_registry_keys_empty() -> None:
    """Test keys() on empty registry."""
    registry = Registry[str, int]()
    assert list(registry.keys()) == []


def test_registry_values() -> None:
    """Test values() method returns all values."""
    registry = Registry[str, int]({"a": 1, "b": 2, "c": 3})
    assert list(registry.values()) == [1, 2, 3]


def test_registry_values_returns_copy() -> None:
    """Test that values() returns a copy, not a view of internal
    state."""
    registry = Registry[str, int]({"a": 1})
    values = list(registry.values())
    registry["b"] = 2
    assert len(values) == 1


def test_registry_values_empty() -> None:
    """Test values() on empty registry."""
    registry = Registry[str, int]()
    assert list(registry.values()) == []


# Test operator overloading


def test_registry_contains_operator() -> None:
    """Test the 'in' operator."""
    registry = Registry[str, int]({"key1": 42})
    assert "key1" in registry


def test_registry_not_contains_operator() -> None:
    """Test the 'not in' operator."""
    registry = Registry[str, int]({"key1": 42})
    assert "missing" not in registry


def test_registry_getitem_operator() -> None:
    """Test the bracket access operator."""
    registry = Registry[str, int]({"key1": 42})
    assert registry["key1"] == 42


def test_registry_getitem_missing_raises_error() -> None:
    """Test that bracket access on missing key raises KeyError."""
    registry = Registry[str, int]()
    with pytest.raises(KeyError, match="Key 'missing' is not registered"):
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
    with pytest.raises(KeyError, match="Key 'missing' is not registered"):
        del registry["missing"]


def test_registry_iter() -> None:
    """Test iterating over registry yields keys."""
    registry = Registry[str, int]({"a": 1, "b": 2, "c": 3})
    assert list(registry) == ["a", "b", "c"]


def test_registry_iter_returns_copy() -> None:
    """Test that iteration uses a snapshot of keys."""
    registry = Registry[str, int]({"a": 1})
    iterator = iter(registry)
    registry["b"] = 2
    keys = list(iterator)
    assert keys == ["a"]


def test_registry_iter_empty() -> None:
    """Test iterating over an empty registry."""
    registry = Registry[str, int]()
    assert list(registry) == []


def test_registry_len_operator() -> None:
    """Test the len() function."""
    assert len(Registry[str, int]()) == 0


def test_registry_len_operator_one_item() -> None:
    registry = Registry[str, int]()
    registry.register("a", 1)
    assert len(registry) == 1


def test_registry_len_operator_two_items() -> None:
    registry = Registry[str, int]()
    registry.register("a", 1)
    registry.register("b", 2)
    assert len(registry) == 2


# Test with different types


def test_registry_with_int_keys() -> None:
    """Test registry with integer keys."""
    registry = Registry[int, str]()
    registry.register(1, "one")
    registry.register(2, "two")
    assert registry.get(1) == "one"


def test_registry_with_int_keys_second_key() -> None:
    registry = Registry[int, str]()
    registry.register(1, "one")
    registry.register(2, "two")
    assert registry.get(2) == "two"


def test_registry_with_tuple_keys() -> None:
    """Test registry with tuple keys."""
    registry = Registry[tuple, str]()
    registry.register((1, 2), "pair")
    assert registry.get((1, 2)) == "pair"


def test_registry_with_complex_values() -> None:
    """Test registry with complex value types."""
    registry = Registry[str, list]()
    registry.register("list1", [1, 2, 3])
    registry.register("list2", ["a", "b", "c"])
    assert registry.get("list1") == [1, 2, 3]


def test_registry_with_complex_values_second_value() -> None:
    registry = Registry[str, list]()
    registry.register("list1", [1, 2, 3])
    registry.register("list2", ["a", "b", "c"])
    assert registry.get("list2") == ["a", "b", "c"]


###########################################
#     Tests for the lazy registration     #
###########################################


def lazy_entries() -> dict[str, int]:
    return {"a": 1}


##################################
#     Tests for the base behavior     #
##################################


def test_registry_register_and_get() -> None:
    """Test basic registration and retrieval on the base class."""
    registry = Registry[str, int]()
    registry.register("key1", 42)
    assert registry.get("key1") == 42


def test_registry_changed_hook_called_on_register() -> None:
    """Test that ``_changed`` is called when a key is registered."""

    class TrackedRegistry(Registry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _changed(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register("key1", 42)
    assert registry.changes == 1


def test_registry_changed_hook_called_on_register_many() -> None:
    """Test that ``_changed`` is called when multiple keys are
    registered."""

    class TrackedRegistry(Registry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _changed(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register_many({"key1": 42, "key2": 100})
    assert registry.changes == 1


def test_registry_register_many_raises_on_duplicate() -> None:
    """Test that a ``register_many`` call (exist_ok=False, a duplicate
    key present) raises an error."""
    registry = Registry[str, int]()
    registry.register("key1", 1)
    with pytest.raises(AlreadyRegisteredError, match=r"key1"):
        registry.register_many({"key1": 100, "key2": 2, "key3": 3})


def test_registry_register_many_is_atomic_per_registry_on_duplicate() -> None:
    """Test that a failed ``register_many`` call (exist_ok=False, a
    duplicate key present) leaves the registry unchanged - i.e. that
    atomicity holds *within a single registry*."""
    registry = Registry[str, int]()
    registry.register("key1", 1)
    with contextlib.suppress(AlreadyRegisteredError):
        registry.register_many({"key1": 100, "key2": 2, "key3": 3})
    # No partial registration occurred: only the pre-existing key is present.
    assert dict(registry.items()) == {"key1": 1}


def test_registry_register_many_is_not_atomic_across_registries() -> None:
    """Test that ``register_many``'s atomicity is per-registry, not
    cross-registry: if the same mapping is registered into several
    registries in sequence, a failure on a later registry does not roll
    back an earlier, already-successful call."""
    mapping = {"key1": 1, "key2": 2}
    registry_a = Registry[str, int]()
    registry_b = Registry[str, int]()
    registry_b.register("key1", 999)  # pre-existing key to force a failure

    registry_a.register_many(mapping)  # succeeds
    with contextlib.suppress(AlreadyRegisteredError):
        registry_b.register_many(mapping)  # fails

    # registry_a's successful registration is not rolled back just because
    # a later registry's registration failed.
    assert dict(registry_a.items()) == mapping


def test_registry_register_many_failure_leaves_the_failing_registry_unchanged() -> None:
    """Test that the failing registry keeps its pre-existing state when
    the same mapping is registered into several registries."""
    mapping = {"key1": 1, "key2": 2}
    registry_a = Registry[str, int]()
    registry_b = Registry[str, int]()
    registry_b.register("key1", 999)  # pre-existing key to force a failure

    registry_a.register_many(mapping)  # succeeds
    with contextlib.suppress(AlreadyRegisteredError):
        registry_b.register_many(mapping)  # fails

    assert dict(registry_b.items()) == {"key1": 999}


def test_registry_changed_hook_called_on_unregister() -> None:
    """Test that ``_changed`` is called when a key is unregistered."""

    class TrackedRegistry(Registry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _changed(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register("key1", 42)
    registry.unregister("key1")
    assert registry.changes == 2  # register + unregister


def test_registry_changed_hook_called_on_clear() -> None:
    """Test that ``_changed`` is called when the registry is cleared."""

    class TrackedRegistry(Registry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _changed(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register("key1", 42)
    registry.clear()
    assert registry.changes == 2  # register + clear


def test_registry_changed_not_called_when_register_fails() -> None:
    """Test that ``_changed`` is not called when a duplicate
    registration raises."""

    class TrackedRegistry(Registry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _changed(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register("key1", 42)
    with contextlib.suppress(AlreadyRegisteredError):
        registry.register("key1", 100)
    assert registry.changes == 1  # only the first successful register


################################################################
#     Tests for the items()/keys()/values() snapshot cache     #
################################################################


def test_registry_iter_yields_the_keys() -> None:
    """Test that ``__iter__`` yields the registered keys."""
    registry = Registry[str, int]({"key1": 1, "key2": 2})
    assert list(registry) == ["key1", "key2"]


def _iterate_and_register(registry: Registry[str, int]) -> list[str]:
    keys = []
    for key in registry:
        keys.append(key)
        registry.register(f"new-{key}", 0)
    return keys


def test_registry_iter_is_stable_when_registry_is_mutated_during_iteration() -> None:
    """Test that mutating the registry while iterating neither raises
    nor changes what the in-flight iterator yields."""
    registry = Registry[str, int]({"key1": 1, "key2": 2})
    assert _iterate_and_register(registry) == ["key1", "key2"]


def test_registry_iter_mutation_during_iteration_is_applied() -> None:
    """Test that the keys registered while iterating are kept."""
    registry = Registry[str, int]({"key1": 1, "key2": 2})
    _iterate_and_register(registry)
    assert len(registry) == 4


def test_registry_items_reflect_the_state_after_register() -> None:
    """Test that ``items`` reflects the new state after ``register``."""
    registry = Registry[str, int]({"key1": 1})
    registry.items()
    registry.register("key2", 2)
    assert dict(registry.items()) == {"key1": 1, "key2": 2}


def test_registry_items_reflect_the_state_after_register_many() -> None:
    """Test that ``items`` reflects the new state after
    ``register_many``."""
    registry = Registry[str, int]({"key1": 1})
    registry.items()
    registry.register_many({"key2": 2, "key3": 3})
    assert dict(registry.items()) == {"key1": 1, "key2": 2, "key3": 3}


def test_registry_items_reflect_the_state_after_unregister() -> None:
    """Test that ``items`` reflects the new state after
    ``unregister``."""
    registry = Registry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    registry.unregister("key1")
    assert dict(registry.items()) == {"key2": 2}


def test_registry_items_reflect_the_state_after_clear() -> None:
    """Test that ``items`` reflects the new state after ``clear``."""
    registry = Registry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    registry.clear()
    assert dict(registry.items()) == {}


def test_registry_snapshot_is_isolated_from_live_state() -> None:
    """Test that the returned views are a detached snapshot: mutating
    the registry afterward does not change a previously returned
    view."""
    registry = Registry[str, int]({"key1": 1})
    items = registry.items()
    registry.register("key2", 2)
    assert dict(items) == {"key1": 1}


def test_registry_keys_snapshot_is_isolated_from_live_state() -> None:
    """Test that the returned keys are a detached snapshot."""
    registry = Registry[str, int]({"key1": 1})
    keys = registry.keys()
    registry.register("key2", 2)
    assert list(keys) == ["key1"]


def test_registry_values_snapshot_is_isolated_from_live_state() -> None:
    """Test that the returned values are a detached snapshot."""
    registry = Registry[str, int]({"key1": 1})
    values = registry.values()
    registry.register("key2", 2)
    assert list(values) == [1]


def test_registry_items_keys_values_consistent_with_each_other() -> None:
    """Test that ``items``/``keys``/``values`` called back-to-back
    (using the same cached snapshot) stay mutually consistent."""
    registry = Registry[str, int]({"key1": 1, "key2": 2})
    assert dict(registry.items()) == {"key1": 1, "key2": 2}


def test_registry_keys_consistent_with_items() -> None:
    """Test that ``keys`` called after ``items`` is consistent."""
    registry = Registry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    assert list(registry.keys()) == ["key1", "key2"]


def test_registry_values_consistent_with_items() -> None:
    """Test that ``values`` called after ``items`` is consistent."""
    registry = Registry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    assert list(registry.values()) == [1, 2]


def test_registry_is_tracked_for_forks() -> None:
    registry = Registry[str, int]()
    assert registry in fork._TRACKED


def test_registry_reset_after_fork_recreates_the_lock() -> None:
    registry = Registry[str, int]()
    lock = registry._lock
    registry._reset_after_fork()
    assert registry._lock is not lock


def test_registry_reset_after_fork_keeps_the_content() -> None:
    registry = Registry[str, int]({"key": 1})
    registry._reset_after_fork()
    assert registry.get("key") == 1


def test_registry_reset_after_fork_does_not_block_if_the_lock_was_held() -> None:
    registry = Registry[str, int]()
    registry._lock.acquire()  # held by a thread that does not exist in the child
    registry._reset_after_fork()
    registry.register("key", 1)
    assert registry.get("key") == 1


def test_registry_copy_returns_the_same_class() -> None:
    assert type(Registry[str, int]().copy()) is Registry


def test_registry_copy_has_the_same_entries() -> None:
    registry = Registry[str, int]({"a": 1, "b": 2})
    assert dict(registry.copy().items()) == {"a": 1, "b": 2}


def test_registry_copy_is_a_new_object() -> None:
    registry = Registry[str, int]()
    assert registry.copy() is not registry


def test_registry_copy_register_does_not_change_the_original() -> None:
    registry = Registry[str, int]({"a": 1})
    registry.copy().register("b", 2)
    assert not registry.has("b")


def test_registry_copy_unregister_does_not_change_the_original() -> None:
    registry = Registry[str, int]({"a": 1})
    registry.copy().unregister("a")
    assert registry.has("a")


def test_registry_copy_clear_does_not_change_the_original() -> None:
    registry = Registry[str, int]({"a": 1})
    registry.copy().clear()
    assert len(registry) == 1


def test_registry_copy_register_in_the_original_does_not_change_the_copy() -> None:
    registry = Registry[str, int]()
    other = registry.copy()
    registry.register("a", 1)
    assert not other.has("a")


def test_registry_copy_does_not_copy_the_values() -> None:
    value = object()
    assert Registry[str, object]({"a": value}).copy().get("a") is value


def test_registry_copy_has_its_own_lock() -> None:
    registry = Registry[str, int]()
    assert registry.copy()._lock is not registry._lock


def test_registry_copy_is_tracked_for_forks() -> None:
    assert Registry[str, int]().copy() in fork._TRACKED


def test_registry_copy_of_a_type_registry_has_its_own_cache() -> None:
    registry = TypeRegistry[str]({int: "int"})
    registry.resolve(int)
    other = registry.copy()
    other.register(bool, "bool")
    assert registry.resolve(bool) == "int"


def test_registry_copy_of_a_type_registry_resolves_in_the_copy() -> None:
    registry = TypeRegistry[str]({int: "int"})
    registry.resolve(bool)
    other = registry.copy()
    other.register(bool, "bool")
    assert other.resolve(bool) == "bool"


def test_registry_items_is_a_snapshot() -> None:
    registry = Registry[str, int]({"a": 1})
    view = registry.items()
    registry.register("b", 2)
    assert list(view) == [("a", 1)]


def test_registry_keys_is_a_snapshot() -> None:
    registry = Registry[str, int]({"a": 1})
    view = registry.keys()
    registry.register("b", 2)
    assert list(view) == ["a"]


def test_registry_values_is_a_snapshot() -> None:
    registry = Registry[str, int]({"a": 1})
    view = registry.values()
    registry.register("b", 2)
    assert list(view) == [1]


def test_registry_iter_is_a_snapshot() -> None:
    registry = Registry[str, int]({"a": 1})
    iterator = iter(registry)
    registry.register("b", 2)
    assert list(iterator) == ["a"]


def test_registry_register_while_iterating_is_safe() -> None:
    registry = Registry[str, int]({"a": 1, "b": 2})
    for key in registry:
        registry.register(f"{key}2", 0)
    assert len(registry) == 4


def test_registry_unregister_does_not_change_an_existing_view() -> None:
    registry = Registry[str, int]({"a": 1, "b": 2})
    view = registry.items()
    registry.unregister("a")
    assert list(view) == [("a", 1), ("b", 2)]


def test_registry_clear_does_not_change_an_existing_view() -> None:
    registry = Registry[str, int]({"a": 1})
    view = registry.items()
    registry.clear()
    assert list(view) == [("a", 1)]


def test_registry_copy_shares_no_write_with_the_source() -> None:
    registry = Registry[str, int]({"a": 1})
    other = registry.copy()
    registry.register("b", 2)
    assert list(other.keys()) == ["a"]


def test_registry_exist_ok_is_keyword_only() -> None:
    registry = Registry[str, int]()
    with pytest.raises(TypeError, match="positional"):
        registry.register("a", 1, True)  # ty: ignore[too-many-positional-arguments]


def test_registry_register_many_exist_ok_is_keyword_only() -> None:
    registry = Registry[str, int]()
    with pytest.raises(TypeError, match="positional"):
        registry.register_many({"a": 1}, True)  # ty: ignore[too-many-positional-arguments]


def test_registry_repr_literal() -> None:
    registry = Registry[str, int]({"a": 1})
    assert repr(registry) == "Registry(\n  (a): 1\n)"


def test_registry_equal_copy_is_equal() -> None:
    registry = Registry[str, int]({"a": 1})
    assert registry.copy().equal(registry)


def test_registry_equal_different_value() -> None:
    assert not Registry[str, int]({"a": 1}).equal(Registry[str, int]({"a": 2}))


def test_registry_equal_different_class() -> None:
    assert not Registry[str, int]({"a": 1}).equal(TypeRegistry[int]())


def test_registry_equal_handlers_are_compared_with_eq() -> None:
    assert Registry[str, list]({"a": [1]}).equal(Registry[str, list]({"a": [1]}))


def test_registry_equal_handlers_without_eq_are_equal_to_themselves_only() -> None:
    assert not Registry[str, object]({"a": object()}).equal(Registry[str, object]({"a": object()}))


def test_registry_equal_after_registration_is_false() -> None:
    registry = Registry[str, int]({"a": 1})
    other = registry.copy()
    other.register("b", 2)
    assert not registry.equal(other)


def test_registry_equal_after_unregistration_is_false() -> None:
    registry = Registry[str, int]({"a": 1})
    other = registry.copy()
    other.unregister("a")
    assert not registry.equal(other)


########################################
#     Tests for pickle and deepcopy    #
########################################


def test_registry_pickle_round_trip_keeps_the_entries() -> None:
    registry = Registry[str, int]({"a": 1, "b": 2})
    assert pickle_round_trip(registry).equal(registry)


def test_registry_pickle_round_trip_keeps_the_class() -> None:
    registry = Registry[str, int]({"a": 1})
    assert type(pickle_round_trip(registry)) is Registry


def test_registry_pickle_clone_can_register() -> None:
    clone = pickle_round_trip(Registry[str, int]({"a": 1}))
    clone.register("b", 2)
    assert clone.get("b") == 2


def test_registry_pickle_clone_has_its_own_lock() -> None:
    registry = Registry[str, int]({"a": 1})
    assert pickle_round_trip(registry)._lock is not registry._lock


def test_registry_pickle_clone_is_tracked_for_forks() -> None:
    assert pickle_round_trip(Registry[str, int]()) in fork._TRACKED


def test_registry_pickle_does_not_change_the_source() -> None:
    registry = Registry[str, int]({"a": 1})
    pickle.dumps(registry)
    assert registry.get("a") == 1


def test_registry_deepcopy_keeps_the_entries() -> None:
    registry = Registry[str, list]({"a": [1]})
    assert copy.deepcopy(registry).equal(registry)


def test_registry_deepcopy_copies_the_values() -> None:
    registry = Registry[str, list]({"a": [1]})
    assert copy.deepcopy(registry).get("a") is not registry.get("a")


def test_registry_deepcopy_has_its_own_lock() -> None:
    registry = Registry[str, int]()
    assert copy.deepcopy(registry)._lock is not registry._lock


def test_registry_copy_module_copy_is_independent() -> None:
    registry = Registry[str, int]({"a": 1})
    other = copy.copy(registry)
    other.register("b", 2)
    assert not registry.has("b")

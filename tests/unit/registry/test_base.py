from __future__ import annotations

import contextlib
from unittest.mock import Mock

import pytest

from coola.registry import AlreadyRegisteredError, Registry, TypeRegistry
from coola.registry.base import BaseRegistry
from coola.utils import fork

##################################
#     Tests for BaseRegistry     #
##################################


def test_registry_is_subclass_of_base_registry() -> None:
    """Test that ``Registry`` reuses the shared base implementation."""
    assert issubclass(Registry, BaseRegistry)


def test_type_registry_is_subclass_of_base_registry() -> None:
    """Test that ``TypeRegistry`` reuses the shared base
    implementation."""
    assert issubclass(TypeRegistry, BaseRegistry)


def test_base_registry_init_empty() -> None:
    """Test creating an empty registry directly from the base class."""
    registry = BaseRegistry[str, int]()
    assert len(registry) == 0


def test_base_registry_register_and_get() -> None:
    """Test basic registration and retrieval on the base class."""
    registry = BaseRegistry[str, int]()
    registry.register("key1", 42)
    assert registry.get("key1") == 42


def test_base_registry_getitem_missing_raises_error() -> None:
    """Test that bracket access on a missing key raises KeyError."""
    registry = BaseRegistry[str, int]()
    with pytest.raises(KeyError, match=r"Key 'missing' is not registered"):
        _ = registry["missing"]


def test_base_registry_unregister_missing_key_raises_error() -> None:
    """Test that unregistering a missing key raises KeyError."""
    registry = BaseRegistry[str, int]()
    with pytest.raises(KeyError, match=r"Key 'missing' is not registered"):
        registry.unregister("missing")


def test_base_registry_on_change_hook_called_on_register() -> None:
    """Test that ``_on_change`` is called when a key is registered."""

    class TrackedRegistry(BaseRegistry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _on_change(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register("key1", 42)
    assert registry.changes == 1


def test_base_registry_on_change_hook_called_on_register_many() -> None:
    """Test that ``_on_change`` is called when multiple keys are
    registered."""

    class TrackedRegistry(BaseRegistry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _on_change(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register_many({"key1": 42, "key2": 100})
    assert registry.changes == 1


def test_base_registry_register_many_raises_on_duplicate() -> None:
    """Test that a ``register_many`` call (exist_ok=False, a duplicate
    key present) raises an error."""
    registry = BaseRegistry[str, int]()
    registry.register("key1", 1)
    with pytest.raises(RuntimeError, match=r"key1"):
        registry.register_many({"key1": 100, "key2": 2, "key3": 3})


def test_base_registry_register_many_is_atomic_per_registry_on_duplicate() -> None:
    """Test that a failed ``register_many`` call (exist_ok=False, a
    duplicate key present) leaves the registry unchanged - i.e. that
    atomicity holds *within a single registry*."""
    registry = BaseRegistry[str, int]()
    registry.register("key1", 1)
    with contextlib.suppress(RuntimeError):
        registry.register_many({"key1": 100, "key2": 2, "key3": 3})
    # No partial registration occurred: only the pre-existing key is present.
    assert dict(registry.items()) == {"key1": 1}


def test_base_registry_register_many_is_not_atomic_across_registries() -> None:
    """Test that ``register_many``'s atomicity is per-registry, not
    cross-registry: if the same mapping is registered into several
    registries in sequence, a failure on a later registry does not roll
    back an earlier, already-successful call."""
    mapping = {"key1": 1, "key2": 2}
    registry_a = BaseRegistry[str, int]()
    registry_b = BaseRegistry[str, int]()
    registry_b.register("key1", 999)  # pre-existing key to force a failure

    registry_a.register_many(mapping)  # succeeds
    with contextlib.suppress(RuntimeError):
        registry_b.register_many(mapping)  # fails

    # registry_a's successful registration is not rolled back just because
    # a later registry's registration failed.
    assert dict(registry_a.items()) == mapping


def test_base_registry_register_many_failure_leaves_the_failing_registry_unchanged() -> None:
    """Test that the failing registry keeps its pre-existing state when
    the same mapping is registered into several registries."""
    mapping = {"key1": 1, "key2": 2}
    registry_a = BaseRegistry[str, int]()
    registry_b = BaseRegistry[str, int]()
    registry_b.register("key1", 999)  # pre-existing key to force a failure

    registry_a.register_many(mapping)  # succeeds
    with contextlib.suppress(RuntimeError):
        registry_b.register_many(mapping)  # fails

    assert dict(registry_b.items()) == {"key1": 999}


def test_base_registry_on_change_hook_called_on_unregister() -> None:
    """Test that ``_on_change`` is called when a key is unregistered."""

    class TrackedRegistry(BaseRegistry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _on_change(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register("key1", 42)
    registry.unregister("key1")
    assert registry.changes == 2  # register + unregister


def test_base_registry_on_change_hook_called_on_clear() -> None:
    """Test that ``_on_change`` is called when the registry is
    cleared."""

    class TrackedRegistry(BaseRegistry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _on_change(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register("key1", 42)
    registry.clear()
    assert registry.changes == 2  # register + clear


def test_base_registry_on_change_not_called_when_register_fails() -> None:
    """Test that ``_on_change`` is not called when a duplicate
    registration raises."""

    class TrackedRegistry(BaseRegistry[str, int]):
        def __init__(self) -> None:
            super().__init__()
            self.changes = 0

        def _on_change(self) -> None:
            self.changes += 1

    registry = TrackedRegistry()
    registry.register("key1", 42)
    with contextlib.suppress(AlreadyRegisteredError):
        registry.register("key1", 100)
    assert registry.changes == 1  # only the first successful register


################################################################
#     Tests for the items()/keys()/values() snapshot cache     #
################################################################


def test_base_registry_items_keys_values_reuse_cached_snapshot() -> None:
    """Test that repeated ``items``/``keys``/``values`` calls between
    mutations reuse the same cached ``dict.copy()`` instead of copying
    ``_state`` again on every call."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    snapshot1 = registry._get_snapshot()
    snapshot2 = registry._get_snapshot()
    assert snapshot1 is snapshot2


def test_base_registry_items_uses_cached_snapshot() -> None:
    """Test that ``items`` is backed by the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    assert registry._snapshot is registry._get_snapshot()


def test_base_registry_keys_uses_cached_snapshot() -> None:
    """Test that ``keys`` is backed by the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.keys()
    assert registry._snapshot is registry._get_snapshot()


def test_base_registry_values_uses_cached_snapshot() -> None:
    """Test that ``values`` is backed by the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.values()
    assert registry._snapshot is registry._get_snapshot()


def test_base_registry_iter_uses_cached_snapshot() -> None:
    """Test that ``__iter__`` is backed by the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    list(registry)
    assert registry._snapshot is registry._get_snapshot()


def test_base_registry_iter_yields_the_keys() -> None:
    """Test that ``__iter__`` yields the registered keys."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    assert list(registry) == ["key1", "key2"]


def test_base_registry_repr_uses_cached_snapshot() -> None:
    """Test that ``__repr__`` is backed by the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    repr(registry)
    assert registry._snapshot is registry._get_snapshot()


def test_base_registry_str_uses_cached_snapshot() -> None:
    """Test that ``__str__`` is backed by the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    str(registry)
    assert registry._snapshot is registry._get_snapshot()


def test_base_registry_repr_does_not_load_the_lazy_groups() -> None:
    """Test that ``__repr__`` does not call the factory of a lazy
    group."""
    factory = Mock(return_value={"key2": 2})
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.register_lazy("group", factory)
    repr(registry)
    factory.assert_not_called()


def test_base_registry_repr_lists_the_lazy_groups() -> None:
    """Test that ``__repr__`` lists the groups that are not loaded."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.register_lazy("group", lambda: {"key2": 2})
    assert repr(registry) == "BaseRegistry(\n  (key1): 1\n  <not loaded: group>\n)"


def test_base_registry_repr_only_lazy_groups() -> None:
    """Test that ``__repr__`` has no empty line when no entry is
    loaded."""
    registry = BaseRegistry[str, int]()
    registry.register_lazy(["b", "a"], lambda: {"key": 1})
    assert repr(registry) == "BaseRegistry(\n  <not loaded: a, b>\n)"


def test_base_registry_repr_after_loading_shows_the_entries() -> None:
    """Test that ``__repr__`` shows the entries of a group once it is
    loaded."""
    registry = BaseRegistry[str, int]()
    registry.register_lazy("group", lambda: {"key": 1})
    len(registry)
    assert repr(registry) == "BaseRegistry(\n  (key): 1\n)"


def test_base_registry_str_does_not_load_the_lazy_groups() -> None:
    """Test that ``__str__`` does not call the factory of a lazy
    group."""
    factory = Mock(return_value={"key2": 2})
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.register_lazy("group", factory)
    str(registry)
    factory.assert_not_called()


def test_base_registry_str_lists_the_lazy_groups() -> None:
    """Test that ``__str__`` lists the groups that are not loaded."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.register_lazy("group", lambda: {"key2": 2})
    assert str(registry) == "BaseRegistry(\n  (key1): 1\n  <not loaded: group>\n)"


def _iterate_and_register(registry: BaseRegistry[str, int]) -> list[str]:
    keys = []
    for key in registry:
        keys.append(key)
        registry.register(f"new-{key}", 0)
    return keys


def test_base_registry_iter_is_stable_when_registry_is_mutated_during_iteration() -> None:
    """Test that mutating the registry while iterating neither raises
    nor changes what the in-flight iterator yields."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    assert _iterate_and_register(registry) == ["key1", "key2"]


def test_base_registry_iter_mutation_during_iteration_is_applied() -> None:
    """Test that the keys registered while iterating are kept."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    _iterate_and_register(registry)
    assert len(registry) == 4


def test_base_registry_snapshot_cache_none_before_first_read() -> None:
    """Test that no snapshot is created until ``items``/``keys``/
    ``values`` is called."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    assert registry._snapshot is None


def test_base_registry_snapshot_cache_invalidated_by_register() -> None:
    """Test that ``register`` invalidates the cached snapshot and that
    ``items`` reflects the new state afterwards."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.items()
    registry.register("key2", 2)
    assert registry._snapshot is None


def test_base_registry_items_reflect_the_state_after_register() -> None:
    """Test that ``items`` reflects the new state after ``register``."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.items()
    registry.register("key2", 2)
    assert dict(registry.items()) == {"key1": 1, "key2": 2}


def test_base_registry_snapshot_cache_is_created_by_items() -> None:
    """Test that ``items`` creates the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.items()
    assert registry._snapshot is not None


def test_base_registry_snapshot_cache_invalidated_by_register_many() -> None:
    """Test that ``register_many`` invalidates the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.items()
    registry.register_many({"key2": 2, "key3": 3})
    assert registry._snapshot is None


def test_base_registry_items_reflect_the_state_after_register_many() -> None:
    """Test that ``items`` reflects the new state after
    ``register_many``."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.items()
    registry.register_many({"key2": 2, "key3": 3})
    assert dict(registry.items()) == {"key1": 1, "key2": 2, "key3": 3}


def test_base_registry_snapshot_cache_invalidated_by_unregister() -> None:
    """Test that ``unregister`` invalidates the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    registry.unregister("key1")
    assert registry._snapshot is None


def test_base_registry_items_reflect_the_state_after_unregister() -> None:
    """Test that ``items`` reflects the new state after
    ``unregister``."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    registry.unregister("key1")
    assert dict(registry.items()) == {"key2": 2}


def test_base_registry_snapshot_cache_invalidated_by_clear() -> None:
    """Test that ``clear`` invalidates the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    registry.clear()
    assert registry._snapshot is None


def test_base_registry_items_reflect_the_state_after_clear() -> None:
    """Test that ``items`` reflects the new state after ``clear``."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    registry.clear()
    assert dict(registry.items()) == {}


def test_base_registry_snapshot_cache_not_invalidated_when_register_fails() -> None:
    """Test that a failed ``register`` call (duplicate key,
    exist_ok=False) does not invalidate an already-cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.items()
    snapshot = registry._snapshot
    with contextlib.suppress(AlreadyRegisteredError):
        registry.register("key1", 2)
    assert registry._snapshot is snapshot


def test_base_registry_snapshot_is_isolated_from_live_state() -> None:
    """Test that the returned views are a detached snapshot: mutating
    the registry afterward does not change a previously returned
    view."""
    registry = BaseRegistry[str, int]({"key1": 1})
    items = registry.items()
    registry.register("key2", 2)
    assert dict(items) == {"key1": 1}


def test_base_registry_keys_snapshot_is_isolated_from_live_state() -> None:
    """Test that the returned keys are a detached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1})
    keys = registry.keys()
    registry.register("key2", 2)
    assert list(keys) == ["key1"]


def test_base_registry_values_snapshot_is_isolated_from_live_state() -> None:
    """Test that the returned values are a detached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1})
    values = registry.values()
    registry.register("key2", 2)
    assert list(values) == [1]


def test_base_registry_items_keys_values_consistent_with_each_other() -> None:
    """Test that ``items``/``keys``/``values`` called back-to-back
    (using the same cached snapshot) stay mutually consistent."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    assert dict(registry.items()) == {"key1": 1, "key2": 2}


def test_base_registry_keys_consistent_with_items() -> None:
    """Test that ``keys`` called after ``items`` is consistent."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    assert list(registry.keys()) == ["key1", "key2"]


def test_base_registry_values_consistent_with_items() -> None:
    """Test that ``values`` called after ``items`` is consistent."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    assert list(registry.values()) == [1, 2]


def test_base_registry_is_tracked_for_forks() -> None:
    registry = BaseRegistry[str, int]()
    assert registry in fork._TRACKED


def test_base_registry_reset_after_fork_recreates_the_lock() -> None:
    registry = BaseRegistry[str, int]()
    lock = registry._lock
    registry._reset_after_fork()
    assert registry._lock is not lock


def test_base_registry_reset_after_fork_keeps_the_content() -> None:
    registry = BaseRegistry[str, int]({"key": 1})
    registry._reset_after_fork()
    assert registry.get("key") == 1


def test_base_registry_reset_after_fork_does_not_block_if_the_lock_was_held() -> None:
    registry = BaseRegistry[str, int]()
    registry._lock.acquire()  # held by a thread that does not exist in the child
    registry._reset_after_fork()
    registry.register("key", 1)
    assert registry.get("key") == 1

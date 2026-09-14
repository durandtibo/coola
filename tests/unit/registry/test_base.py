from __future__ import annotations

import pytest

from coola.registry import Registry, TypeRegistry
from coola.registry.base import BaseRegistry

###################################
#     Tests for BaseRegistry     #
###################################


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


def test_base_registry_register_many_is_atomic_per_registry_on_duplicate() -> None:
    """Test that a failed ``register_many`` call (exist_ok=False, a
    duplicate key present) leaves the registry unchanged - i.e. that
    atomicity holds *within a single registry*."""
    registry = BaseRegistry[str, int]()
    registry.register("key1", 1)
    with pytest.raises(RuntimeError, match=r"key1"):
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
    with pytest.raises(RuntimeError, match=r"key1"):
        registry_b.register_many(mapping)  # fails

    # registry_a's successful registration is not rolled back just because
    # a later registry's registration failed.
    assert dict(registry_a.items()) == mapping
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
    with pytest.raises(RuntimeError):
        registry.register("key1", 100)
    assert registry.changes == 1  # only the first successful register


#############################################################
#     Tests for the items()/keys()/values() snapshot cache     #
#############################################################


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
    assert registry._snapshot is not None
    registry.register("key2", 2)
    assert registry._snapshot is None
    assert dict(registry.items()) == {"key1": 1, "key2": 2}


def test_base_registry_snapshot_cache_invalidated_by_register_many() -> None:
    """Test that ``register_many`` invalidates the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.items()
    registry.register_many({"key2": 2, "key3": 3})
    assert registry._snapshot is None
    assert dict(registry.items()) == {"key1": 1, "key2": 2, "key3": 3}


def test_base_registry_snapshot_cache_invalidated_by_unregister() -> None:
    """Test that ``unregister`` invalidates the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    registry.unregister("key1")
    assert registry._snapshot is None
    assert dict(registry.items()) == {"key2": 2}


def test_base_registry_snapshot_cache_invalidated_by_clear() -> None:
    """Test that ``clear`` invalidates the cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    registry.items()
    registry.clear()
    assert registry._snapshot is None
    assert dict(registry.items()) == {}


def test_base_registry_snapshot_cache_not_invalidated_when_register_fails() -> None:
    """Test that a failed ``register`` call (duplicate key,
    exist_ok=False) does not invalidate an already-cached snapshot."""
    registry = BaseRegistry[str, int]({"key1": 1})
    registry.items()
    snapshot = registry._snapshot
    with pytest.raises(RuntimeError):
        registry.register("key1", 2)
    assert registry._snapshot is snapshot


def test_base_registry_snapshot_is_isolated_from_live_state() -> None:
    """Test that the returned views are a detached snapshot: mutating
    the registry afterward does not change a previously returned
    view."""
    registry = BaseRegistry[str, int]({"key1": 1})
    items = registry.items()
    keys = registry.keys()
    values = registry.values()
    registry.register("key2", 2)
    assert dict(items) == {"key1": 1}
    assert list(keys) == ["key1"]
    assert list(values) == [1]


def test_base_registry_items_keys_values_consistent_with_each_other() -> None:
    """Test that ``items``/``keys``/``values`` called back-to-back
    (using the same cached snapshot) stay mutually consistent."""
    registry = BaseRegistry[str, int]({"key1": 1, "key2": 2})
    assert dict(registry.items()) == {"key1": 1, "key2": 2}
    assert list(registry.keys()) == ["key1", "key2"]
    assert list(registry.values()) == [1, 2]

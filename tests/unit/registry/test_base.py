from __future__ import annotations

import threading

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


def test_base_registry_thread_safety_concurrent_register() -> None:
    """Test that concurrent registrations do not corrupt the registry
    state."""
    registry = BaseRegistry[int, int]()

    def register_range(start: int, end: int) -> None:
        for i in range(start, end):
            registry.register(i, i * 2)

    threads = [
        threading.Thread(target=register_range, args=(i * 100, (i + 1) * 100)) for i in range(10)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(registry) == 1000
    assert registry.get(500) == 1000

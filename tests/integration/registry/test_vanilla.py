from __future__ import annotations

import threading
from collections import Counter
from typing import TYPE_CHECKING

from coola.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Sequence


def run_threads(threads: Sequence[threading.Thread]) -> None:
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


##############################
#     Tests for Registry     #
##############################


def test_registry_full_workflow() -> None:
    """Test a complete workflow with multiple operations."""
    registry = Registry[str, int]()

    # Register multiple items
    registry.register("a", 1)
    registry.register("b", 2)
    registry["c"] = 3

    # Check state
    assert len(registry) == 3
    assert "a" in registry

    # Update existing
    registry["b"] = 20
    assert registry.get("b") == 20

    # Unregister
    value = registry.unregister("a")
    assert value == 1
    assert len(registry) == 2

    # Clear all
    registry.clear()
    assert len(registry) == 0


def test_registry_concurrent_register_different_keys() -> None:
    """Test that multiple threads can register different keys
    simultaneously."""
    registry = Registry[str, int]()

    def register_key(index: int) -> None:
        registry.register(f"key_{index}", index)

    run_threads([threading.Thread(target=register_key, args=(i,)) for i in range(10)])

    # Verify all keys were registered
    assert registry.equal(
        Registry[str, int](
            {
                "key_0": 0,
                "key_1": 1,
                "key_2": 2,
                "key_3": 3,
                "key_4": 4,
                "key_5": 5,
                "key_6": 6,
                "key_7": 7,
                "key_8": 8,
                "key_9": 9,
            }
        )
    )


def test_registry_concurrent_register_same_key_with_exist_ok() -> None:
    """Test that multiple threads can safely overwrite the same key with
    exist_ok=True."""
    registry = Registry[str, int]()
    num_threads = 10

    def register_shared_key(value: int) -> None:
        registry.register("key", value, exist_ok=True)

    run_threads(
        [threading.Thread(target=register_shared_key, args=(i,)) for i in range(num_threads)]
    )

    # The key should exist with one of the values
    assert registry.has("key")
    assert 0 <= registry.get("key") < num_threads


def test_registry_concurrent_register_same_key_without_exist_ok() -> None:
    """Test that concurrent registration of same key without exist_ok
    raises errors correctly."""
    registry = Registry[str, int]()
    num_threads = 10
    errors = []
    successes = []

    def register_with_error_handling(value: int) -> None:
        try:
            registry.register("key", value, exist_ok=False)
            successes.append(value)
        except RuntimeError:
            errors.append(value)

    run_threads(
        [
            threading.Thread(target=register_with_error_handling, args=(i,))
            for i in range(num_threads)
        ]
    )

    # Exactly one thread should succeed, others should fail
    assert len(successes) == 1
    assert len(errors) == num_threads - 1
    assert registry.has("key")
    assert 0 <= registry.get("key") < num_threads


def test_registry_concurrent_get() -> None:
    """Test that multiple threads can read the same key
    simultaneously."""
    registry = Registry[str, int]()
    registry.register("key", 42)
    num_threads = 10
    results = []

    def read_key() -> None:
        value = registry.get("key")
        results.append(value)

    run_threads([threading.Thread(target=read_key) for i in range(num_threads)])

    # All threads should have read the correct value
    assert len(results) == num_threads
    assert all(value == 42 for value in results)


def test_registry_concurrent_has() -> None:
    """Test that multiple threads can check key existence
    simultaneously."""
    registry = Registry[str, int]()
    registry.register("existing_key", 100)
    num_threads = 10
    results = Counter()

    def check_keys() -> None:
        exists = registry.has("existing_key")
        not_exists = registry.has("missing_key")
        results["exists"] += 1 if exists else 0
        results["not_exists"] += 1 if not not_exists else 0

    run_threads([threading.Thread(target=check_keys) for i in range(num_threads)])

    assert results["exists"] == num_threads
    assert results["not_exists"] == num_threads


def test_registry_concurrent_unregister_operations() -> None:
    """Test that concurrent unregister operations handle conflicts
    correctly."""
    num_keys = 10
    registry = Registry[str, int]({f"key_{i}": i for i in range(num_keys)})

    threads = []
    unregistered_values = []
    errors = []

    def unregister_key(key: str) -> None:
        try:
            value = registry.unregister(key)
            unregistered_values.append(value)
        except KeyError:
            errors.append(key)

    # Try to unregister each key from two threads
    for i in range(num_keys):
        thread1 = threading.Thread(target=unregister_key, args=(f"key_{i}",))
        thread2 = threading.Thread(target=unregister_key, args=(f"key_{i}",))
        threads.extend([thread1, thread2])
        thread1.start()
        thread2.start()

    for thread in threads:
        thread.join()

    # Each key should be unregistered exactly once
    assert len(unregistered_values) == num_keys
    assert len(errors) == num_keys
    assert len(registry) == 0


def test_registry_concurrent_clear_operations() -> None:
    """Test that concurrent clear operations don't cause issues."""
    registry = Registry[str, int]({f"key_{i}": i for i in range(20)})
    num_threads = 10

    def clear_registry() -> None:
        registry.clear()

    run_threads([threading.Thread(target=clear_registry) for i in range(num_threads)])

    assert len(registry) == 0
    assert registry.equal(Registry[str, int]())


def test_registry_concurrent_register_many() -> None:
    """Test that register_many is thread-safe."""
    registry = Registry[str, int]()
    num_threads = 10

    def register_batch(offset: int) -> None:
        batch = {f"key_{offset}_{i}": offset * 100 + i for i in range(10)}
        registry.register_many(batch)

    run_threads([threading.Thread(target=register_batch, args=(i,)) for i in range(num_threads)])

    # All keys should be registered
    assert len(registry) == num_threads * 10


def test_registry_concurrent_mixed_operations() -> None:
    """Test a realistic scenario with mixed read/write operations."""
    registry = Registry[str, int]({f"initial_{i}": i for i in range(10)})
    num_threads = 10

    def mixed_operations(thread_id: int) -> None:
        # Register new keys
        registry.register(f"thread_{thread_id}", thread_id, exist_ok=True)

        # Read existing keys
        if registry.has(f"initial_{thread_id % 10}"):
            registry.get(f"initial_{thread_id % 10}")

        # Update existing keys
        registry[f"thread_{thread_id}"] = thread_id * 2

    run_threads([threading.Thread(target=mixed_operations, args=(i,)) for i in range(num_threads)])

    # Verify final state
    assert len(registry) >= 10  # At least the initial keys
    for i in range(num_threads):
        assert registry.has(f"thread_{i}")


def test_registry_concurrent_dict_style_operations() -> None:
    """Test thread-safety of dictionary-style operations."""
    registry = Registry[str, int]()
    num_threads = 10

    def dict_operations(index: int) -> None:
        # Set using bracket notation
        registry[f"key_{index}"] = index

        # Get using bracket notation
        if f"key_{index}" in registry:
            value = registry[f"key_{index}"]
            assert value == index

        # Update
        registry[f"key_{index}"] = index * 2

    run_threads([threading.Thread(target=dict_operations, args=(i,)) for i in range(num_threads)])

    assert len(registry) == num_threads
    for i in range(num_threads):
        assert registry[f"key_{i}"] == i * 2


def test_registry_concurrent_equal_operations() -> None:
    """Test that equal() is thread-safe."""
    registry1 = Registry[str, int]({"a": 1, "b": 2})
    registry2 = Registry[str, int]({"a": 1, "b": 2})
    num_threads = 10
    results = []

    def check_equality() -> None:
        result = registry1.equal(registry2)
        results.append(result)

    run_threads([threading.Thread(target=check_equality) for i in range(num_threads)])

    assert len(results) == num_threads
    assert all(result is True for result in results)


def test_registry_stress_test_high_contention() -> None:
    """Stress test with high contention on a single key."""
    registry = Registry[str, int]()
    num_threads = 100
    counter = {"value": 0}
    lock = threading.Lock()

    def stress_operations(thread_id: int) -> None:
        for i in range(10):
            # Mix of operations
            key = f"key_{i % 2}"
            registry.register(key, thread_id, exist_ok=True)
            if registry.has(key):
                registry.get(key)
            with lock:
                counter["value"] += 1

    run_threads([threading.Thread(target=stress_operations, args=(i,)) for i in range(num_threads)])

    # Just verify no crashes occurred and operations completed
    assert counter["value"] == num_threads * 10
    assert len(registry) > 0

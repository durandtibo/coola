from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from coola.registry import TypeRegistry
from tests.integration.registry.test_vanilla import run_threads


class Animal:
    """Base animal class for testing inheritance."""


class Dog(Animal):
    """Dog class inheriting from Animal."""


class Cat(Animal):
    """Cat class inheriting from Animal."""


class Poodle(Dog):
    """Poodle class inheriting from Dog."""


def create_classes(num_classes: int) -> list[type]:
    out = []
    for _ in range(num_classes):

        class MyClass: ...

        out.append(MyClass)
    return out


##################################
#     Tests for TypeRegistry     #
##################################


def test_type_registry_full_workflow() -> None:
    """Test a complete workflow with multiple operations."""
    registry = TypeRegistry[str]()

    # Register multiple items
    registry.register(int, "integer")
    registry.register(float, "float")
    registry[str] = "string"

    # Check state
    assert len(registry) == 3
    assert int in registry

    # Update existing
    registry[float] = "float32"
    assert registry.get(float) == "float32"

    # Unregister
    value = registry.unregister(int)
    assert value == "integer"
    assert len(registry) == 2

    # Clear all
    registry.clear()
    assert len(registry) == 0


def test_type_registry_concurrent_register_different_keys() -> None:
    """Test that multiple threads can register different keys
    simultaneously."""
    registry = TypeRegistry[int]()
    num_threads = 10
    classes = create_classes(num_classes=num_threads)

    def register_key(index: int) -> None:
        registry.register(classes[index], index)

    run_threads([threading.Thread(target=register_key, args=(i,)) for i in range(num_threads)])

    # Verify all keys were registered
    assert registry.equal(
        TypeRegistry[int](
            {
                classes[0]: 0,
                classes[1]: 1,
                classes[2]: 2,
                classes[3]: 3,
                classes[4]: 4,
                classes[5]: 5,
                classes[6]: 6,
                classes[7]: 7,
                classes[8]: 8,
                classes[9]: 9,
            }
        )
    )


def test_type_registry_concurrent_register_same_key_with_exist_ok() -> None:
    """Test that multiple threads can safely overwrite the same key with
    exist_ok=True."""
    registry = TypeRegistry[int]()
    num_threads = 10

    def register_shared_key(index: int) -> None:
        registry.register(int, index, exist_ok=True)

    run_threads(
        [threading.Thread(target=register_shared_key, args=(i,)) for i in range(num_threads)]
    )

    # The key should exist with one of the values
    assert registry.has(int)
    assert 0 <= registry.get(int) < num_threads


def test_type_registry_concurrent_register_same_key_without_exist_ok() -> None:
    """Test that concurrent registration of same key without exist_ok
    raises errors correctly."""
    registry = TypeRegistry[int]()
    num_threads = 10
    errors = []
    successes = []

    def register_with_error_handling(value: int) -> None:
        try:
            registry.register(int, value, exist_ok=False)
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
    assert registry.has(int)
    assert 0 <= registry.get(int) < num_threads


def test_type_registry_concurrent_get() -> None:
    """Test that multiple threads can read the same key
    simultaneously."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    num_threads = 10
    results = []

    def read_key() -> None:
        value = registry.get(int)
        results.append(value)

    run_threads([threading.Thread(target=read_key) for _ in range(num_threads)])

    # All threads should have read the correct value
    assert len(results) == num_threads
    assert all(value == "integer" for value in results)


def test_type_registry_concurrent_getitem() -> None:
    """Test that multiple threads can read the same key
    simultaneously."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    num_threads = 10
    results = []

    def read_key() -> None:
        value = registry[int]
        results.append(value)

    run_threads([threading.Thread(target=read_key) for _ in range(num_threads)])

    # All threads should have read the correct value
    assert len(results) == num_threads
    assert all(value == "integer" for value in results)


def test_type_registry_concurrent_has() -> None:
    """Test that multiple threads can check key existence
    simultaneously."""
    registry = TypeRegistry[str]()
    registry.register(int, "integer")
    num_threads = 10
    lock = threading.Lock()
    results = defaultdict(int)

    def check_keys() -> None:
        exists = registry.has(int)
        not_exists = registry.has(float)
        with lock:
            results["exists"] += 1 if exists else 0
            results["not_exists"] += 1 if not not_exists else 0

    run_threads([threading.Thread(target=check_keys) for _ in range(num_threads)])

    assert results["exists"] == num_threads
    assert results["not_exists"] == num_threads


def test_type_registry_concurrent_unregister_operations() -> None:
    """Test that concurrent unregister operations handle conflicts
    correctly."""
    num_classes = 10
    classes = create_classes(num_classes=num_classes)
    registry = TypeRegistry[int]({classes[i]: i for i in range(num_classes)})

    unregistered_values = []
    errors = []

    def unregister_type(dtype: type) -> None:
        try:
            value = registry.unregister(dtype)
            unregistered_values.append(value)
        except KeyError:
            errors.append(dtype)

    # Try to unregister each key from two threads
    threads = []
    for i in range(num_classes):
        thread1 = threading.Thread(target=unregister_type, args=(classes[i],))
        thread2 = threading.Thread(target=unregister_type, args=(classes[i],))
        threads.extend([thread1, thread2])

    run_threads(threads)

    # Each key should be unregistered exactly once
    assert len(unregistered_values) == num_classes
    assert len(errors) == num_classes
    assert len(registry) == 0


def test_type_registry_concurrent_clear_operations() -> None:
    """Test that concurrent clear operations don't cause issues."""
    registry = TypeRegistry[str]({int: "integer", float: "float", str: "string"})
    num_threads = 10

    def clear_registry() -> None:
        registry.clear()

    run_threads([threading.Thread(target=clear_registry) for _ in range(num_threads)])

    assert len(registry) == 0
    assert registry.equal(TypeRegistry[str]())


def test_type_registry_concurrent_register_many() -> None:
    """Test that register_many is thread-safe."""
    registry = TypeRegistry[str]()
    num_threads = 10

    def register_batch() -> None:
        class A: ...

        class B: ...

        class C: ...

        batch = {A: "a", B: "b", C: "c"}
        registry.register_many(batch)

    run_threads([threading.Thread(target=register_batch) for _ in range(num_threads)])

    # All keys should be registered
    assert len(registry) == num_threads * 3


def test_type_registry_concurrent_mixed_operations() -> None:
    """Test a realistic scenario with mixed read/write operations."""
    registry = TypeRegistry[int]()
    num_threads = 10
    classes = create_classes(num_classes=num_threads)

    def mixed_operations(thread_id: int) -> None:
        # Register new keys
        registry.register(classes[thread_id], thread_id, exist_ok=True)

        # Read existing keys
        if registry.has(classes[thread_id]):
            registry.get(classes[thread_id])

        # Update existing keys
        registry[classes[thread_id]] = thread_id * 2

    run_threads([threading.Thread(target=mixed_operations, args=(i,)) for i in range(num_threads)])

    # Verify final state
    assert len(registry) == num_threads
    for i in range(num_threads):
        assert registry.has(classes[i])


def test_type_registry_concurrent_dict_style_operations() -> None:
    """Test thread-safety of dictionary-style operations."""
    registry = TypeRegistry[int]()
    num_threads = 10
    classes = create_classes(num_classes=num_threads)

    def dict_operations(index: int) -> None:
        # Set using bracket notation
        registry[classes[index]] = index

        # Get using bracket notation
        if classes[index] in registry:
            value = registry[classes[index]]
            assert value == index

        # Update
        registry[classes[index]] = index * 2

    run_threads([threading.Thread(target=dict_operations, args=(i,)) for i in range(num_threads)])

    assert len(registry) == num_threads
    for i in range(num_threads):
        assert registry[classes[i]] == i * 2


def test_type_registry_concurrent_equal_operations() -> None:
    """Test that equal() is thread-safe."""
    registry1 = TypeRegistry[str]({int: "integer", float: "float", str: "string"})
    registry2 = TypeRegistry[str]({int: "integer", float: "float", str: "string"})
    num_threads = 10
    results = []

    def check_equality() -> None:
        result = registry1.equal(registry2)
        results.append(result)

    run_threads([threading.Thread(target=check_equality) for _ in range(num_threads)])

    assert len(results) == num_threads
    assert all(result is True for result in results)


def test_type_registry_concurrent_resolve_same_type() -> None:
    """Test that concurrent resolve calls for the same type are thread-
    safe."""
    registry = TypeRegistry[str]({object: "base", int: "integer"})
    num_threads = 10
    results = []
    errors = []

    def resolve_type() -> None:
        try:
            result = registry.resolve(int)
            results.append(result)
        except Exception as e:
            errors.append(e)

    run_threads([threading.Thread(target=resolve_type) for _ in range(num_threads)])

    assert not errors, f"Errors occurred: {errors}"
    assert len(results) == num_threads
    assert all(r == "integer" for r in results)


def test_type_registry_concurrent_resolve_different_types() -> None:
    """Test thread safety when resolving different types
    concurrently."""
    registry = TypeRegistry[str]({Animal: "animal", Dog: "dog", Cat: "cat"})
    num_threads = 100
    results = defaultdict(list)
    lock = threading.Lock()

    def resolve_type(dtype: type, expected: type) -> None:
        result = registry.resolve(dtype)
        with lock:
            results[dtype].append((result, expected))

    # Execute concurrent resolves with mixed types
    threads = []
    for i in range(num_threads):
        if i % 3 == 0:
            t = threading.Thread(target=resolve_type, args=(Dog, "dog"))
        elif i % 3 == 1:
            t = threading.Thread(target=resolve_type, args=(Cat, "cat"))
        else:
            t = threading.Thread(target=resolve_type, args=(Animal, "animal"))
        threads.append(t)

    run_threads(threads)

    for result_list in results.values():
        for result, expected in result_list:
            assert result == expected


def test_type_registry_cache_consistency_under_concurrent_access() -> None:
    """Test that cache remains consistent under concurrent access."""
    registry = TypeRegistry[str]({object: "base", Dog: "dog"})
    num_threads = 100
    cache_results = []

    def resolve_and_check() -> None:
        # First resolution (might trigger cache population)
        first = registry.resolve(Poodle)
        time.sleep(0.0001)  # Small delay to encourage race conditions
        # Second resolution (should use cache)
        second = registry.resolve(Poodle)
        cache_results.append((first, second))

    # Execute
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(resolve_and_check) for _ in range(num_threads)]
        for future in as_completed(futures):
            future.result()  # Raise any exceptions

    # Assert
    assert len(cache_results) == num_threads
    for first, second in cache_results:
        assert first == "dog"
        assert second == "dog"
        assert first == second


def test_type_registry_concurrent_resolve_with_cache_miss() -> None:
    """Test thread safety when multiple threads trigger cache miss
    simultaneously."""
    registry = TypeRegistry[str]({Animal: "animal", Dog: "dog"})
    num_threads = 10
    results = []
    barrier = threading.Barrier(num_threads)  # Synchronize thread start

    def resolve_with_barrier() -> None:
        barrier.wait()  # All threads start at once
        result = registry.resolve(Poodle)
        results.append(result)

    run_threads([threading.Thread(target=resolve_with_barrier) for _ in range(num_threads)])

    assert len(results) == num_threads
    assert all(r == "dog" for r in results)


def test_type_registry_no_race_condition_on_cache_population() -> None:
    """Test that cache population doesn't create race conditions."""
    registry = TypeRegistry[str]({object: "base", Animal: "animal"})
    types_to_resolve = [Dog, Cat, Poodle, bool, str, int]
    num_iterations = 5
    all_results = {dtype: [] for dtype in types_to_resolve}
    lock = threading.Lock()

    def resolve_all_types() -> None:
        for dtype in types_to_resolve:
            result = registry.resolve(dtype)
            with lock:
                all_results[dtype].append(result)

    # Execute multiple iterations
    for _ in range(num_iterations):
        run_threads([threading.Thread(target=resolve_all_types) for _ in range(10)])

    # Assert - all resolutions of same type should return same result
    for results in all_results.values():
        assert len(set(results)) == 1


def test_type_registry_stress_test_high_contention() -> None:
    """Stress test with high contention on a single key."""
    registry = TypeRegistry[int]()
    num_threads = 100
    counter = defaultdict(int)
    lock = threading.Lock()
    classes = create_classes(num_classes=num_threads)

    def stress_operations(thread_id: int) -> None:
        for i in range(10):
            # Mix of operations
            key = classes[i]
            registry.register(key, thread_id, exist_ok=True)
            if registry.has(key):
                registry.get(key)
            with lock:
                counter["value"] += 1

    run_threads([threading.Thread(target=stress_operations, args=(i,)) for i in range(num_threads)])

    # Just verify no crashes occurred and operations completed
    assert counter["value"] == num_threads * 10
    assert len(registry) > 0

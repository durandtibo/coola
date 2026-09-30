from __future__ import annotations

import threading

from coola.registry.base import BaseRegistry
from tests.integration.helpers import run_threads

###################################
#     Tests for BaseRegistry     #
###################################


def test_base_registry_thread_safety_concurrent_register() -> None:
    """Test that concurrent registrations do not corrupt the registry
    state."""
    registry = BaseRegistry[int, int]()

    def register_range(start: int, end: int) -> None:
        for i in range(start, end):
            registry.register(i, i * 2)

    run_threads(
        [threading.Thread(target=register_range, args=(i * 100, (i + 1) * 100)) for i in range(10)]
    )

    assert len(registry) == 1000
    assert registry.get(500) == 1000


def test_base_registry_concurrent_mutation_stress() -> None:
    """Test that concurrent register/unregister/read calls stay
    consistent."""
    registry = BaseRegistry[int, int]()
    num_threads, num_keys = 8, 200
    errors: list[BaseException] = []
    barrier = threading.Barrier(num_threads)

    def mutate(thread_id: int) -> None:
        try:
            barrier.wait()
            for i in range(num_keys):
                key = thread_id * num_keys + i
                registry.register(key, key)
                assert registry[key] == key
                list(registry.items())  # iterate while other threads mutate
                if i % 2:
                    assert registry.unregister(key) == key
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    run_threads([threading.Thread(target=mutate, args=(i,)) for i in range(num_threads)])

    assert not errors
    assert len(registry) == num_threads * num_keys // 2
    assert all((key % num_keys) % 2 == 0 for key in registry)

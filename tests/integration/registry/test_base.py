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

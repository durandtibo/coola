from __future__ import annotations

import threading

from coola.hashing import HasherRegistry, StrHasher
from tests.integration.helpers import run_threads


def is_never(data_type: type) -> bool:  # noqa: ARG001
    return False


####################################
#     Tests for HasherRegistry     #
####################################


def test_hasher_registry_register_fallback_thread_safe() -> None:
    """Test that concurrent calls to register_fallback() do not lose a
    registration."""
    registry = HasherRegistry()
    hashers = [StrHasher() for _ in range(200)]

    def _worker(hasher: StrHasher) -> None:
        registry.register_fallback(is_never, hasher)

    run_threads([threading.Thread(target=_worker, args=(hasher,)) for hasher in hashers])

    assert len(registry._fallbacks) == len(hashers)

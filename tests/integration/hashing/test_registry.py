from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from coola.hashing import HasherRegistry, StrHasher
from tests.integration.helpers import run_threads

if TYPE_CHECKING:
    from collections.abc import Callable


####################################
#     Tests for HasherRegistry     #
####################################


def test_hasher_registry_register_fallback_thread_safe() -> None:
    """Test that concurrent calls to register_fallback() do not lose a
    registration."""
    registry = HasherRegistry()
    predicates = [lambda _, i=i: i < 0 for i in range(200)]

    def _worker(predicate: Callable[[type], bool]) -> None:
        registry.register_fallback(predicate, StrHasher())

    run_threads([threading.Thread(target=_worker, args=(p,)) for p in predicates])

    assert len(registry.fallbacks()) == len(predicates)

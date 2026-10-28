from __future__ import annotations

import threading

from coola.identifier import generate_uuid7
from tests.integration.helpers import run_threads

####################################
#     Tests for generate_uuid7     #
####################################


def test_generate_uuid7_thread_safe_unique() -> None:
    """Test that concurrent calls never produce duplicate
    identifiers."""
    ids: list[str] = []
    lock = threading.Lock()
    num_threads = 8

    def _worker() -> None:
        local = [generate_uuid7() for _ in range(1000)]
        with lock:
            ids.extend(local)

    run_threads([threading.Thread(target=_worker) for _ in range(num_threads)])

    assert len(ids) == num_threads * 1000
    assert len(ids) == len(set(ids))


def test_generate_uuid7_thread_safe_increasing_per_thread() -> None:
    """Test that the identifiers of each thread increase."""
    results: list[bool] = []
    lock = threading.Lock()

    def _worker() -> None:
        local = [generate_uuid7() for _ in range(1000)]
        with lock:
            results.append(local == sorted(local))

    run_threads([threading.Thread(target=_worker) for _ in range(8)])

    assert results == [True] * 8

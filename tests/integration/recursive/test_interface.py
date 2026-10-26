from __future__ import annotations

import threading
from typing import Any

from coola.recursive import recursive_apply
from tests.integration.helpers import run_threads

NUM_THREADS = 8


def test_recursive_apply_concurrent_calls_do_not_share_state() -> None:
    """Test that concurrent calls of ``recursive_apply`` on the default
    registry return the right result for their own data."""
    results: list[Any] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        for _ in range(50):
            result = recursive_apply({"a": [index, (index, {index})], "b": index}, lambda x: x + 1)
            with lock:
                results.append(result)

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)])
    assert len(results) == NUM_THREADS * 50
    assert {r["b"] for r in results} == set(range(1, NUM_THREADS + 1))
    assert all(r["a"] == [r["b"], (r["b"], {r["b"]})] for r in results)


def test_recursive_apply_concurrent_calls_detect_cycles() -> None:
    """Test that a cycle in one thread does not affect the calls of the
    other threads."""
    errors: list[Exception] = []
    results: list[Any] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        data: list[Any] = [index]
        if index % 2:
            data.append(data)
        try:
            result = recursive_apply(data, str)
        except ValueError as exc:
            with lock:
                errors.append(exc)
        else:
            with lock:
                results.append(result)

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(NUM_THREADS)])
    assert len(errors) == NUM_THREADS // 2
    assert len(results) == NUM_THREADS // 2

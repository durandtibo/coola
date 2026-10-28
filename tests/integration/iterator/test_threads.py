from __future__ import annotations

import threading
from typing import Any

import pytest

from coola.iterator import bfs_iterate, dfs_items, dfs_iterate
from tests.integration.helpers import run_threads

NUM_THREADS = 8


@pytest.mark.parametrize("iterate", [dfs_iterate, bfs_iterate])
def test_iterate_in_threads_with_the_default_registry(iterate: Any) -> None:
    """Test that threads can traverse data with the default registry at
    the same time."""
    results: list[list[int]] = []
    lock = threading.Lock()

    def worker(offset: int) -> None:
        leaves = sorted(iterate({"a": [offset, [offset + 1]], "b": (offset + 2,)}))
        with lock:
            results.append(leaves)

    run_threads([threading.Thread(target=worker, args=(i * 10,)) for i in range(NUM_THREADS)])

    assert sorted(results) == [[i * 10, i * 10 + 1, i * 10 + 2] for i in range(NUM_THREADS)]


@pytest.mark.parametrize("iterate", [dfs_iterate, bfs_iterate])
def test_iterate_in_threads_reports_a_cycle_in_each_thread(iterate: Any) -> None:
    """Test that a cycle is reported in every thread."""
    errors: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        data: list = [1]
        data.append(data)
        try:
            list(iterate(data))
        except ValueError as exc:
            with lock:
                errors.append(str(exc))

    run_threads([threading.Thread(target=worker) for _ in range(NUM_THREADS)])

    assert len(errors) == NUM_THREADS


def test_dfs_items_in_threads_with_the_default_registry() -> None:
    """Test that threads can traverse data with their paths and the
    default registry at the same time."""
    results: list[list[tuple[tuple[Any, ...], int]]] = []
    lock = threading.Lock()

    def worker(offset: int) -> None:
        items = list(dfs_items({"a": [offset, [offset + 1]], "b": (offset + 2,)}))
        with lock:
            results.append(items)

    run_threads([threading.Thread(target=worker, args=(i * 10,)) for i in range(NUM_THREADS)])

    assert sorted(results) == [
        [(("a", 0), i * 10), (("a", 1, 0), i * 10 + 1), (("b", 0), i * 10 + 2)]
        for i in range(NUM_THREADS)
    ]

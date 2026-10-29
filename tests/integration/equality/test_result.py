from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from coola.equality import ComparisonResult, compare
from tests.integration.helpers import run_threads

if TYPE_CHECKING:
    import pytest

LOGGER_NAME = "coola.equality"

#############################
#     Tests for compare     #
#############################


def test_compare_concurrent_threads() -> None:
    """Test that concurrent comparisons each report their own
    difference."""
    num_threads = 4
    results: dict[int, ComparisonResult] = {}

    def worker(index: int) -> None:
        for _ in range(20):
            results[index] = compare({"k": [0, index]}, {"k": [0, index + 1]})

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(num_threads)])

    for i in range(num_threads):
        assert results[i] == ComparisonResult(
            equal=False,
            path=("k", 1),
            reason=f"numbers are different:\n  actual   : {i}\n  expected : {i + 1}",
        )


def test_compare_concurrent_calls_do_not_mix_results() -> None:
    """Test that concurrent comparisons report their own path and that
    equal objects are never reported as different."""
    num_threads = 8
    barrier = threading.Barrier(num_threads)
    errors: list[str] = []

    def worker(index: int) -> None:
        barrier.wait()
        for _ in range(50):
            result = compare({"key": [index, 1]}, {"key": [index, 2]})
            if result.path != ("key", 1):
                errors.append(f"worker {index}: unexpected path {result.path}")
            if not compare({"key": index}, {"key": index}):
                errors.append(f"worker {index}: equal objects reported as different")

    run_threads([threading.Thread(target=worker, args=(i,)) for i in range(num_threads)])

    assert not errors


def test_compare_does_not_drop_logs_of_other_threads(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a log record emitted by another thread while a
    comparison is running is not dropped."""
    started = threading.Event()
    release = threading.Event()

    class Blocking:
        def __eq__(self, other: object) -> bool:
            started.set()
            release.wait(timeout=5)
            return True

        def __hash__(self) -> int:
            return 0

    thread = threading.Thread(target=compare, args=(Blocking(), Blocking()))
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        thread.start()
        assert started.wait(timeout=5)
        logging.getLogger(LOGGER_NAME).info("message from another thread")
        release.set()
        thread.join()

    assert "message from another thread" in caplog.messages

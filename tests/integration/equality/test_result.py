from __future__ import annotations

import threading

from coola.equality import ComparisonResult, compare
from tests.integration.helpers import run_threads

##############################
#     Tests for compare      #
##############################


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
        actual, expected = {"k": [0, i]}, {"k": [0, i + 1]}
        assert results[i] == ComparisonResult(
            equal=False,
            path=("k", 1),
            reason=f"numbers are different:\n  actual   : {i}\n  expected : {i + 1}",
            actual=repr(actual),
            expected=repr(expected),
        )

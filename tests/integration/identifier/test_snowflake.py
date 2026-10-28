from __future__ import annotations

import itertools
import threading

from coola.identifier import SnowflakeIdGenerator
from tests.integration.helpers import run_threads

##########################################
#     Tests for SnowflakeIdGenerator     #
##########################################


def test_snowflake_id_generator_generate_thread_safe_unique() -> None:
    """Test that concurrent calls to generate() never produce duplicate
    identifiers."""
    generator = SnowflakeIdGenerator(worker_id=1)
    ids: list[int] = []
    lock = threading.Lock()
    num_threads = 8

    def _worker() -> None:
        local = [generator.generate() for _ in range(1000)]
        with lock:
            ids.extend(local)

    run_threads([threading.Thread(target=_worker) for _ in range(num_threads)])

    assert len(ids) == num_threads * 1000
    assert len(ids) == len(set(ids))


def test_snowflake_id_generator_generate_thread_safe_increasing_per_thread() -> None:
    """Test that the identifiers of each thread increase."""
    generator = SnowflakeIdGenerator(worker_id=1)
    results: list[bool] = []
    lock = threading.Lock()

    def _worker() -> None:
        local = [generator.generate() for _ in range(1000)]
        with lock:
            results.append(local == sorted(local))

    run_threads([threading.Thread(target=_worker) for _ in range(8)])

    assert results == [True] * 8


def test_snowflake_id_generator_generate_strictly_increases_above_4096_per_millisecond() -> None:
    """Test that 200,000 identifiers from a loop (more than 4,096 per
    millisecond) are strictly increasing and never raise."""
    generator = SnowflakeIdGenerator(worker_id=1)
    ids = [generator.generate() for _ in range(200_000)]
    assert all(a < b for a, b in itertools.pairwise(ids))

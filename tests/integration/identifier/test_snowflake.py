from __future__ import annotations

import threading
import time

from coola.identifier import SnowflakeIdGenerator
from coola.identifier.snowflake import _MAX_SEQUENCE
from tests.integration.helpers import run_threads

##########################################
#     Tests for SnowflakeIdGenerator     #
##########################################


def test_snowflake_id_generator_generate_thread_safe() -> None:
    """Test that concurrent calls to generate() never produce duplicate
    identifiers."""
    generator = SnowflakeIdGenerator()
    ids: list[int] = []
    lock = threading.Lock()
    num_threads = 200

    def _worker() -> None:
        snowflake_id = generator.generate()
        with lock:
            ids.append(snowflake_id)

    run_threads([threading.Thread(target=_worker) for _ in range(num_threads)])

    assert len(ids) == num_threads
    assert len(ids) == len(set(ids))


def test_snowflake_id_generator_generate_thread_safe_during_sequence_rollover() -> None:
    """Test that concurrent calls racing through an exhausted per-
    millisecond sequence never produce duplicate identifiers."""
    generator = SnowflakeIdGenerator()
    generator._last_timestamp_ms = time.time_ns() // 1_000_000
    generator._sequence = _MAX_SEQUENCE
    ids: list[int] = []
    lock = threading.Lock()
    num_threads = 8

    def _worker() -> None:
        snowflake_id = generator.generate()
        with lock:
            ids.append(snowflake_id)

    run_threads([threading.Thread(target=_worker) for _ in range(num_threads)])

    assert len(ids) == num_threads
    assert len(ids) == len(set(ids))

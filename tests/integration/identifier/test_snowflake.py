from __future__ import annotations

import threading
import time
from unittest.mock import patch

from coola.identifier import SnowflakeIdGenerator, snowflake as module
from coola.identifier.snowflake import _MAX_SEQUENCE
from coola.utils.singleton import LazySingleton
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
    generator = SnowflakeIdGenerator(
        last_timestamp_ms=time.time_ns() // 1_000_000, sequence=_MAX_SEQUENCE
    )
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


def test_snowflake_id_generator_generate_thread_safe_when_the_sequence_wraps_repeatedly() -> None:
    """Test that concurrent calls never produce duplicate identifiers
    when the per-millisecond sequence wraps several times."""
    generator = SnowflakeIdGenerator()
    ids: list[int] = []
    lock = threading.Lock()
    num_threads = 8
    num_ids_per_thread = 2_000

    def _worker() -> None:
        local_ids = [generator.generate() for _ in range(num_ids_per_thread)]
        with lock:
            ids.extend(local_ids)

    run_threads([threading.Thread(target=_worker) for _ in range(num_threads)])

    assert len(ids) == num_threads * num_ids_per_thread
    assert len(ids) == len(set(ids))


###########################################
#     Tests for get_default_generator     #
###########################################


def test_get_default_generator_thread_safe() -> None:
    """Test that concurrent first calls build a single generator."""
    built: list[SnowflakeIdGenerator] = []

    def _factory() -> SnowflakeIdGenerator:
        time.sleep(0.01)
        generator = module.SnowflakeIdGenerator()
        built.append(generator)
        return generator

    generators: list[SnowflakeIdGenerator] = []
    lock = threading.Lock()
    num_threads = 8

    def _worker() -> None:
        generator = module.get_default_generator()
        with lock:
            generators.append(generator)

    with patch.object(module, "_default_generator", LazySingleton(_factory)):
        run_threads([threading.Thread(target=_worker) for _ in range(num_threads)])

    assert len(built) == 1
    assert len(generators) == num_threads
    assert all(generator is built[0] for generator in generators)


def test_snowflake_id_generator_generate_fills_the_sequence_within_one_millisecond() -> None:
    """Test that a full sequence of IDs minted within the same
    millisecond is consecutive."""
    generator = SnowflakeIdGenerator()
    fixed_ms = 1_800_000_000_000
    with patch("time.time_ns", return_value=fixed_ms * 1_000_000):
        ids = [generator.generate() for _ in range(_MAX_SEQUENCE + 1)]
    assert ids[-1] == ids[0] + _MAX_SEQUENCE


def test_snowflake_id_generator_generate_exhausts_the_sequence_with_unique_ids() -> None:
    """Test that the whole sequence of a millisecond gives unique
    identifiers."""
    generator = SnowflakeIdGenerator()
    with patch("time.time_ns", return_value=1_800_000_000_000 * 1_000_000):
        ids = {generator.generate() for _ in range(_MAX_SEQUENCE + 1)}
    assert len(ids) == _MAX_SEQUENCE + 1

from __future__ import annotations

import threading
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import coola.identifier.snowflake as snowflake_module
from coola.identifier.snowflake import generate_snowflake_id

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_generator_state() -> Generator[None, None, None]:
    """Reset the module-level generator state before and after each
    test, so tests do not depend on the previous test's timing."""
    snowflake_module._last_timestamp_ms = -1
    snowflake_module._sequence = 0
    yield
    snowflake_module._last_timestamp_ms = -1
    snowflake_module._sequence = 0


######################################
#     Tests for generate_snowflake_id     #
######################################


def test_generate_snowflake_id_returns_int() -> None:
    assert isinstance(generate_snowflake_id(), int)


def test_generate_snowflake_id_is_non_negative() -> None:
    assert generate_snowflake_id() >= 0


def test_generate_snowflake_id_fits_in_64_bits() -> None:
    assert generate_snowflake_id() < 2**63


def test_generate_snowflake_id_is_unique_across_calls() -> None:
    ids = {generate_snowflake_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_generate_snowflake_id_is_monotonically_increasing() -> None:
    previous = generate_snowflake_id()
    for _ in range(1000):
        current = generate_snowflake_id()
        assert current > previous
        previous = current


def test_generate_snowflake_id_same_millisecond_increments_sequence() -> None:
    with patch("time.time_ns", return_value=1_800_000_000_000 * 1_000_000):
        first = generate_snowflake_id()
        second = generate_snowflake_id()
    assert second == first + 1


def test_generate_snowflake_id_default_worker_id_is_zero() -> None:
    snowflake_id = generate_snowflake_id()
    worker_id = (
        snowflake_id >> snowflake_module._WORKER_ID_SHIFT
    ) & snowflake_module._MAX_WORKER_ID
    assert worker_id == 0


def test_generate_snowflake_id_encodes_given_worker_id() -> None:
    snowflake_id = generate_snowflake_id(worker_id=7)
    worker_id = (
        snowflake_id >> snowflake_module._WORKER_ID_SHIFT
    ) & snowflake_module._MAX_WORKER_ID
    assert worker_id == 7


def test_generate_snowflake_id_different_worker_ids_differ() -> None:
    with patch("time.time_ns", return_value=1_800_000_000_000 * 1_000_000):
        assert generate_snowflake_id(worker_id=0) != generate_snowflake_id(worker_id=1)


def test_generate_snowflake_id_negative_worker_id_raises() -> None:
    with pytest.raises(ValueError, match="worker_id must fit in 10 bits"):
        generate_snowflake_id(worker_id=-1)


def test_generate_snowflake_id_worker_id_too_large_raises() -> None:
    with pytest.raises(ValueError, match="worker_id must fit in 10 bits"):
        generate_snowflake_id(worker_id=1024)


def test_generate_snowflake_id_max_worker_id_is_valid() -> None:
    assert isinstance(generate_snowflake_id(worker_id=1023), int)


def test_generate_snowflake_id_clock_moved_backward_raises() -> None:
    snowflake_module._last_timestamp_ms = 9_999_999_999_999
    with pytest.raises(RuntimeError, match="clock moved backward"):
        generate_snowflake_id()


def test_generate_snowflake_id_encodes_timestamp_roundtrip() -> None:
    fixed_ms = 1_800_000_000_000
    with patch("time.time_ns", return_value=fixed_ms * 1_000_000):
        snowflake_id = generate_snowflake_id()
    decoded_ms = (snowflake_id >> snowflake_module._TIMESTAMP_SHIFT) + snowflake_module._EPOCH_MS
    assert decoded_ms == fixed_ms


def test_generate_snowflake_id_sequence_rollover_advances_to_next_millisecond() -> None:
    first_ms = 1_800_000_000_000
    next_ms = first_ms + 1
    snowflake_module._last_timestamp_ms = first_ms
    snowflake_module._sequence = snowflake_module._MAX_SEQUENCE
    # The main body sees `first_ms` again (sequence wraps to 0, forcing
    # the busy-wait loop), which itself observes `first_ms` once more
    # before the clock advances to `next_ms`.
    with patch(
        "time.time_ns",
        side_effect=[first_ms * 1_000_000, first_ms * 1_000_000, next_ms * 1_000_000],
    ):
        snowflake_id = generate_snowflake_id()
    decoded_ms = (snowflake_id >> snowflake_module._TIMESTAMP_SHIFT) + snowflake_module._EPOCH_MS
    assert decoded_ms == next_ms
    assert (snowflake_id & snowflake_module._MAX_SEQUENCE) == 0


def test_generate_snowflake_id_sequence_wraps_within_same_millisecond() -> None:
    fixed_ms = 1_800_000_000_000
    with patch("time.time_ns", return_value=fixed_ms * 1_000_000):
        first = generate_snowflake_id()
        current = first
        for _ in range(snowflake_module._MAX_SEQUENCE):
            current = generate_snowflake_id()
        assert current == first + snowflake_module._MAX_SEQUENCE


def test_generate_snowflake_id_thread_safe() -> None:
    ids: list[int] = []
    lock = threading.Lock()

    def _worker() -> None:
        snowflake_id = generate_snowflake_id()
        with lock:
            ids.append(snowflake_id)

    threads = [threading.Thread(target=_worker) for _ in range(200)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == len(set(ids))

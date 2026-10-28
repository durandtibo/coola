from __future__ import annotations

import itertools
import threading
from unittest.mock import patch

import pytest

from coola.identifier.sequence import TickSequence
from tests.integration.helpers import run_threads

TIME_NS = "coola.identifier.sequence.time.time_ns"


def at(ms: int) -> int:
    return ms * 1_000_000


#####################################
#     Tests for TickSequence     #
#####################################


def _generate_in_the_same_millisecond() -> list[tuple[int, int]]:
    generator = TickSequence(bits=80)
    with patch(TIME_NS, return_value=at(1000)):
        return [generator.next() for _ in range(100)]


def test_tick_sequence_same_timestamp_in_the_same_millisecond() -> None:
    values = _generate_in_the_same_millisecond()
    assert [timestamp for timestamp, _ in values] == [1000] * 100


def test_tick_sequence_increases_in_the_same_millisecond() -> None:
    values = _generate_in_the_same_millisecond()
    assert values == sorted(values)


def test_tick_sequence_increments_the_random_part_in_the_same_millisecond() -> None:
    randoms = [r for _, r in _generate_in_the_same_millisecond()]
    assert all(b == a + 1 for a, b in itertools.pairwise(randoms))


def _generate_in_two_milliseconds() -> tuple[tuple[int, int], tuple[int, int]]:
    generator = TickSequence(bits=80)
    with patch(TIME_NS, return_value=at(1000)):
        first = generator.next()
    with patch(TIME_NS, return_value=at(1001)):
        second = generator.next()
    return first, second


def test_tick_sequence_new_millisecond_has_the_new_timestamp() -> None:
    _, second = _generate_in_two_milliseconds()
    assert second[0] == 1001


def test_tick_sequence_new_millisecond_increases() -> None:
    first, second = _generate_in_two_milliseconds()
    assert second > first


def test_tick_sequence_new_millisecond_gets_a_start_part() -> None:
    first, second = _generate_in_two_milliseconds()
    assert second[1] != first[1] + 1  # the probability of a collision is 2**-80


def test_tick_sequence_clock_moved_backward_keeps_increasing() -> None:
    generator = TickSequence(bits=80)
    with patch(TIME_NS, return_value=at(1000)):
        first = generator.next()
    with patch(TIME_NS, return_value=at(990)):
        second = generator.next()
    assert second == (1000, first[1] + 1)


def _generate_with_an_overflow() -> list[tuple[int, int]]:
    generator = TickSequence(bits=2)
    with patch(TIME_NS, return_value=at(1000)):
        return [generator.next() for _ in range(20)]


def test_tick_sequence_overflow_increases() -> None:
    values = _generate_with_an_overflow()
    assert values == sorted(values)


def test_tick_sequence_overflow_is_unique() -> None:
    assert len(set(_generate_with_an_overflow())) == 20


def test_tick_sequence_overflow_borrows_the_next_millisecond() -> None:
    values = _generate_with_an_overflow()
    assert values[-1][0] > 1000  # more than 4 values needed more than one millisecond


def test_tick_sequence_overflow_keeps_the_random_part_in_range() -> None:
    values = _generate_with_an_overflow()
    assert all(0 <= r < 4 for _, r in values)


def test_tick_sequence_random_part_fits_in_the_bits() -> None:
    generator = TickSequence(bits=74)
    with patch(TIME_NS, side_effect=range(at(1), at(2000), at(1))):
        assert all(generator.next()[1] < 2**74 for _ in range(500))


def test_tick_sequence_random_part_is_random() -> None:
    generator = TickSequence(bits=80)
    with patch(TIME_NS, side_effect=range(at(1), at(100), at(1))):
        randoms = {generator.next()[1] for _ in range(50)}
    assert len(randoms) == 50


def _reset_after_fork() -> tuple[TickSequence, threading.Lock, tuple[int, int]]:
    generator = TickSequence(bits=80)
    lock = generator._lock
    with patch(TIME_NS, return_value=at(1000)):
        generator.next()
        lock.acquire()  # held by a thread that does not exist in the child
        generator._reset_after_fork()
        # a fresh random part is used (not the previous one + 1) and it does not block
        second = generator.next()
    return generator, lock, second


def test_tick_sequence_reset_after_fork_resets_the_last_timestamp() -> None:
    generator = TickSequence(bits=80)
    with patch(TIME_NS, return_value=at(1000)):
        generator.next()
    generator._reset_after_fork()
    assert generator._tick == -1


def test_tick_sequence_reset_after_fork_replaces_the_lock() -> None:
    generator, lock, _ = _reset_after_fork()
    assert generator._lock is not lock


def test_tick_sequence_reset_after_fork_does_not_block() -> None:
    _, _, second = _reset_after_fork()
    assert second[0] == 1000


@pytest.mark.parametrize("bits", [1, 8, 74, 80])
def test_tick_sequence_max_random_is_bits_wide(bits: int) -> None:
    assert TickSequence(bits=bits)._max == 2**bits - 1


def test_tick_sequence_thread_safe_count() -> None:
    generator = TickSequence(bits=80)
    values: list[tuple[int, int]] = []
    lock = threading.Lock()

    def worker() -> None:
        local = [generator.next() for _ in range(200)]
        with lock:
            values.extend(local)

    run_threads([threading.Thread(target=worker) for _ in range(8)])
    assert len(values) == 1600


def test_tick_sequence_thread_safe_uniqueness() -> None:
    generator = TickSequence(bits=80)
    values: list[tuple[int, int]] = []
    lock = threading.Lock()

    def worker() -> None:
        local = [generator.next() for _ in range(200)]
        with lock:
            values.extend(local)

    run_threads([threading.Thread(target=worker) for _ in range(8)])
    assert len(set(values)) == 1600  # no duplicates


def test_tick_sequence_without_random_start_restarts_at_zero() -> None:
    generator = TickSequence(bits=12, random_start=False)
    with patch(TIME_NS, return_value=at(1000)):
        assert generator.next() == (1000, 0)


def test_tick_sequence_without_random_start_increments_in_the_same_millisecond() -> None:
    generator = TickSequence(bits=12, random_start=False)
    with patch(TIME_NS, return_value=at(1000)):
        generator.next()
        assert generator.next() == (1000, 1)


def test_tick_sequence_without_random_start_restarts_at_zero_in_a_new_millisecond() -> None:
    generator = TickSequence(bits=12, random_start=False)
    with patch(TIME_NS, return_value=at(1000)):
        generator.next()
    with patch(TIME_NS, return_value=at(1001)):
        assert generator.next() == (1001, 0)


def test_tick_sequence_without_random_start_borrows_the_next_millisecond() -> None:
    generator = TickSequence(bits=1, random_start=False)
    with patch(TIME_NS, return_value=at(1000)):
        values = [generator.next() for _ in range(3)]
    assert values == [(1000, 0), (1000, 1), (1001, 0)]


def test_tick_sequence_overflow_with_a_random_start_borrows_the_next_tick() -> None:
    generator = TickSequence(bits=1)
    with (
        patch(TIME_NS, return_value=at(1000)),
        patch.object(generator, "_start", return_value=1),
    ):
        values = [generator.next() for _ in range(2)]
    assert values == [(1000, 1), (1001, 1)]


def test_tick_sequence_tick_ns_divides_the_clock() -> None:
    generator = TickSequence(bits=12, tick_ns=1_000_000_000, random_start=False)
    with patch(TIME_NS, return_value=1_700_000_000_999_999_999):
        assert generator.next() == (1_700_000_000, 0)


def test_tick_sequence_tick_ns_keeps_the_tick_within_a_second() -> None:
    generator = TickSequence(bits=12, tick_ns=1_000_000_000, random_start=False)
    with patch(TIME_NS, return_value=1_700_000_000_000_000_000):
        generator.next()
    with patch(TIME_NS, return_value=1_700_000_000_999_999_999):
        assert generator.next() == (1_700_000_000, 1)


def test_tick_sequence_sequences_are_independent() -> None:
    first = TickSequence(bits=12, random_start=False)
    second = TickSequence(bits=12, random_start=False)
    with patch(TIME_NS, return_value=at(1000)):
        first.next()
        first.next()
        assert second.next() == (1000, 0)

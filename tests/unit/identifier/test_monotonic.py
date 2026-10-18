from __future__ import annotations

import itertools
import threading
from unittest.mock import patch

import pytest

from coola.identifier.monotonic import MonotonicRandom
from tests.integration.helpers import run_threads

TIME_NS = "coola.identifier.monotonic.time.time_ns"


def at(ms: int) -> int:
    return ms * 1_000_000


#####################################
#     Tests for MonotonicRandom     #
#####################################


def test_monotonic_random_increases_in_the_same_millisecond() -> None:
    generator = MonotonicRandom(bits=80)
    with patch(TIME_NS, return_value=at(1000)):
        values = [generator.next() for _ in range(100)]
    assert [timestamp for timestamp, _ in values] == [1000] * 100
    assert values == sorted(values)
    randoms = [r for _, r in values]
    assert all(b == a + 1 for a, b in itertools.pairwise(randoms))


def test_monotonic_random_new_millisecond_gets_a_fresh_random_part() -> None:
    generator = MonotonicRandom(bits=80)
    with patch(TIME_NS, return_value=at(1000)):
        first = generator.next()
    with patch(TIME_NS, return_value=at(1001)):
        second = generator.next()
    assert second[0] == 1001
    assert second > first
    assert second[1] != first[1] + 1  # the probability of a collision is 2**-80


def test_monotonic_random_clock_moved_backward_keeps_increasing() -> None:
    generator = MonotonicRandom(bits=80)
    with patch(TIME_NS, return_value=at(1000)):
        first = generator.next()
    with patch(TIME_NS, return_value=at(990)):
        second = generator.next()
    assert second == (1000, first[1] + 1)


def test_monotonic_random_overflow_borrows_the_next_millisecond() -> None:
    generator = MonotonicRandom(bits=2)
    with patch(TIME_NS, return_value=at(1000)):
        values = [generator.next() for _ in range(20)]
    assert values == sorted(values)
    assert len(set(values)) == 20
    assert values[-1][0] > 1000  # more than 4 values needed more than one millisecond
    assert all(0 <= r < 4 for _, r in values)


def test_monotonic_random_random_part_fits_in_the_bits() -> None:
    generator = MonotonicRandom(bits=74)
    with patch(TIME_NS, side_effect=range(at(1), at(2000), at(1))):
        assert all(generator.next()[1] < 2**74 for _ in range(500))


def test_monotonic_random_random_part_is_random() -> None:
    generator = MonotonicRandom(bits=80)
    with patch(TIME_NS, side_effect=range(at(1), at(100), at(1))):
        randoms = {generator.next()[1] for _ in range(50)}
    assert len(randoms) == 50


def test_monotonic_random_reset_after_fork() -> None:
    generator = MonotonicRandom(bits=80)
    lock = generator._lock
    with patch(TIME_NS, return_value=at(1000)):
        generator.next()
        lock.acquire()  # held by a thread that does not exist in the child
        generator._reset_after_fork()
        assert generator._lock is not lock
        assert generator._last_timestamp_ms == -1
        # a fresh random part is used (not the previous one + 1) and it does not block
        second = generator.next()
    assert second[0] == 1000


@pytest.mark.parametrize("bits", [1, 8, 74, 80])
def test_monotonic_random_max_random_is_bits_wide(bits: int) -> None:
    assert MonotonicRandom(bits=bits)._max == 2**bits - 1


def test_monotonic_random_thread_safe() -> None:
    generator = MonotonicRandom(bits=80)
    values: list[tuple[int, int]] = []
    lock = threading.Lock()

    def worker() -> None:
        local = [generator.next() for _ in range(200)]
        with lock:
            values.extend(local)

    run_threads([threading.Thread(target=worker) for _ in range(8)])
    assert len(values) == 1600
    assert len(set(values)) == 1600  # no duplicates

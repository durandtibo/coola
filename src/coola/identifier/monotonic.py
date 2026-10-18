r"""Generate the timestamp and the random part of time-ordered
identifiers that increase within the same millisecond."""

from __future__ import annotations

__all__ = ["MonotonicRandom"]

import os
import threading
import time

from coola.identifier.fork import track_fork


class MonotonicRandom:
    r"""Generate ``(timestamp, random)`` pairs that strictly increase.

    A time-ordered identifier (ULID, UUIDv7) is made of a millisecond
    timestamp followed by random bits. Two identifiers generated in the
    same millisecond would be ordered randomly. This class makes them
    increase: in a new millisecond, the random part is a fresh random
    number; in the same millisecond (or if the clock moved backward),
    the timestamp is kept and the random part of the previous call is
    incremented by one. If the increment overflows, the timestamp is
    advanced by one millisecond with a fresh random part.

    The state is per process and protected by a lock. It is reset in
    the child process of a fork, so the parent and its children do not
    continue the same sequence.

    Args:
        bits: The number of random bits.

    Example:
        ```pycon
        >>> from coola.identifier.monotonic import MonotonicRandom
        >>> generator = MonotonicRandom(bits=80)
        >>> first = generator.next()
        >>> second = generator.next()
        >>> first < second
        True

        ```
    """

    def __init__(self, bits: int) -> None:
        self._bits = bits
        self._max = (1 << bits) - 1
        self._lock = threading.Lock()
        self._last_timestamp_ms = -1
        self._last_random = 0
        track_fork(self)

    def next(self) -> tuple[int, int]:
        r"""Generate the next timestamp and random part.

        Returns:
            A ``(timestamp_ms, random)`` pair, greater than the pair
                returned by the previous call of this generator.
        """
        with self._lock:
            now_ms = time.time_ns() // 1_000_000
            if now_ms > self._last_timestamp_ms:
                self._last_timestamp_ms = now_ms
                self._last_random = self._fresh_random()
            elif self._last_random < self._max:
                self._last_random += 1
            else:  # the random part overflows: borrow the next millisecond
                self._last_timestamp_ms += 1
                self._last_random = self._fresh_random()
            return self._last_timestamp_ms, self._last_random

    def _fresh_random(self) -> int:
        r"""Generate a random number of ``bits`` bits."""
        return int.from_bytes(os.urandom((self._bits + 7) // 8), byteorder="big") & self._max

    def _reset_after_fork(self) -> None:
        # The lock may have been held by a thread that does not exist in the child.
        self._lock = threading.Lock()
        self._last_timestamp_ms = -1
        self._last_random = 0

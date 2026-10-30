r"""Generate the strictly increasing ``(tick, counter)`` pairs of the
time-ordered identifiers."""

from __future__ import annotations

__all__ = ["TickSequence"]

import secrets
import threading
import time

from coola.utils.fork import track_fork


class TickSequence:
    r"""Generate ``(tick, counter)`` pairs that strictly increase.

    A time-ordered identifier (ULID, UUIDv7, ObjectId, Snowflake) is
    made of a tick of the clock (a millisecond, or a second) followed
    by a counter. This class is the one clock rule of the package, and
    it never raises or blocks:

    - in a new tick, the counter restarts (with a random number, or with
      ``0`` if ``random_start`` is ``False``);
    - in the same tick, or if the clock moved backward, the tick is
      kept and the counter of the previous call is incremented by one;
    - if the counter is exhausted, the tick is advanced by one with a
      restarted counter.

    The tick may therefore run ahead of the clock (after a burst or a
    backward step of the clock) until the clock catches up: order and
    uniqueness win over accuracy.

    The state is per object and protected by a lock. It is reset in the
    child process of a fork, so the parent and its children do not
    continue the same sequence.

    Args:
        bits: The number of bits of the counter.
        tick_ns: The length of a tick in nanoseconds. The default is
            a millisecond.
        random_start: If ``True``, the counter restarts with a random
            number in a new tick, so that two processes do not mint the
            same identifiers. If ``False``, it restarts at ``0`` (the
            uniqueness must then come from another field, for example
            the worker id of a Snowflake ID).

    Example:
        ```pycon
        >>> from coola.identifier.sequence import TickSequence
        >>> sequence = TickSequence(bits=80)
        >>> first = sequence.next()
        >>> second = sequence.next()
        >>> first < second
        True

        ```
    """

    def __init__(self, *, bits: int, tick_ns: int = 1_000_000, random_start: bool = True) -> None:
        self._bits = bits
        self._max = (1 << bits) - 1
        self._tick_ns = tick_ns
        self._random_start = random_start
        self._lock = threading.Lock()
        self._tick = -1
        self._counter = 0
        track_fork(self)

    def next(self) -> tuple[int, int]:
        r"""Generate the next tick and counter.

        Returns:
            A ``(tick, counter)`` pair, greater than the pair
                returned by the previous call of this object.
        """
        with self._lock:
            now = time.time_ns() // self._tick_ns
            if now > self._tick:  # a new tick: the counter restarts
                self._tick = now
                self._counter = self._start()
            elif self._counter < self._max:  # the same tick, or the clock moved backward
                self._counter += 1
            else:  # the counter is exhausted: borrow the next tick
                self._tick += 1
                self._counter = self._start()
            return self._tick, self._counter

    def _start(self) -> int:
        r"""Generate the first counter of a tick.

        Returns:
            The first counter of the tick.
        """
        return secrets.randbits(self._bits) if self._random_start else 0

    def _reset_after_fork(self) -> None:
        r"""Reset the state in the child process of a fork.

        The lock may have been held by a thread that does not exist in
        the child, so a new one is created.
        """
        # The lock may have been held by a thread that does not exist in the child.
        self._lock = threading.Lock()
        self._tick = -1
        self._counter = 0

r"""Provide a Snowflake-style identifier for time-ordered records.

Like ``generate_ulid``, a Snowflake ID is not derived from data content:
two calls produce different values even for identical input. It packs a
millisecond timestamp, a fixed worker ID, and a per-worker sequence
number into a single 64-bit integer, in the spirit of Twitter's original
Snowflake service. Prefer it over ``generate_ulid`` when the identifier
must be a single 64-bit integer (e.g. a database ``BIGINT`` primary key)
rather than a 26-character string, or when IDs must be attributable to
the worker/shard that minted them.
"""

from __future__ import annotations

__all__ = [
    "SnowflakeIdGenerator",
    "extract_snowflake_timestamp_ms",
    "generate_snowflake_id",
    "get_default_generator",
]

import threading
import time

from coola.identifier.validation import validate_bit_range

# Sleep briefly between spins while waiting for the next millisecond, to
# avoid busy-waiting a full CPU core (and holding the lock) at 100%
# utilization when the per-millisecond sequence is exhausted.
_SPIN_SLEEP_S = 0.0001


# Custom epoch (2024-01-01T00:00:00Z, in ms since the Unix epoch) so the
# 41-bit timestamp field does not waste bits on years before this
# library existed. Shifts the field's effective range to ~69 years from
# this date.
_EPOCH_MS = 1_704_067_200_000

_TIMESTAMP_BITS = 41
_WORKER_ID_BITS = 10
_SEQUENCE_BITS = 12

_MAX_WORKER_ID = (1 << _WORKER_ID_BITS) - 1
_MAX_SEQUENCE = (1 << _SEQUENCE_BITS) - 1

_WORKER_ID_SHIFT = _SEQUENCE_BITS
_TIMESTAMP_SHIFT = _SEQUENCE_BITS + _WORKER_ID_BITS


class SnowflakeIdGenerator:
    r"""Generate Snowflake-style 64-bit identifiers.

    All the state needed to mint monotonically increasing IDs (the
    last timestamp seen and the current sequence number) lives on the
    instance rather than at module scope, so each generator is
    independent: create one per worker/shard/test instead of sharing
    mutable global state.

    Args:
        last_timestamp_ms: The millisecond timestamp of the last ID
            minted by this generator, or ``-1`` (default) if none has
            been minted yet. Pass the value persisted from a previous
            instance (e.g. across a process restart) together with
            ``sequence`` to preserve the monotonically increasing
            guarantee; leave at the default for a fresh generator.
        sequence: The sequence number of the last ID minted for
            ``last_timestamp_ms``. Ignored (treated as ``0``) when
            ``last_timestamp_ms`` is ``-1``.

    Raises:
        ValueError: If ``last_timestamp_ms`` is not ``-1`` and does
            not fit in 41 bits, or if ``sequence`` does not fit in 12
            bits.

    Example:
        ```pycon
        >>> from coola.identifier import SnowflakeIdGenerator
        >>> generator = SnowflakeIdGenerator()
        >>> snowflake_id = generator.generate()
        >>> isinstance(snowflake_id, int)
        True

        ```
    """

    def __init__(self, last_timestamp_ms: int = -1, sequence: int = 0) -> None:
        if last_timestamp_ms != -1:
            # last_timestamp_ms is an absolute Unix ms timestamp (like
            # time.time_ns() // 1_000_000), not one already shifted by
            # _EPOCH_MS, so it must fit in 41 bits only once the epoch
            # is subtracted back out, matching what `generate` does
            # when packing the returned integer.
            validate_bit_range(
                last_timestamp_ms - _EPOCH_MS, _TIMESTAMP_BITS, name="last_timestamp_ms - epoch"
            )
        validate_bit_range(sequence, _SEQUENCE_BITS, name="sequence")
        self._lock = threading.Lock()
        self._last_timestamp_ms = last_timestamp_ms
        self._sequence = sequence

    def generate(self, worker_id: int = 0, timestamp_ms: int | None = None) -> int:
        r"""Generate a Snowflake-style 64-bit identifier.

        The returned integer is composed of a 41-bit millisecond
        timestamp (relative to a fixed epoch), a 10-bit ``worker_id``,
        and a 12-bit sequence number that increments for IDs minted
        within the same millisecond by this generator. Because the
        timestamp is the most significant part, IDs generated later
        are numerically greater than IDs generated earlier (from the
        same, or an earlier, millisecond).

        Note:
            The sequence counter is local to this generator instance:
            it guarantees uniqueness for calls made on this instance
            for a given ``worker_id``, not across other instances or
            processes. Assign each concurrently running generator (one
            per process or shard, typically) a distinct ``worker_id``
            to avoid collisions between them.

        Args:
            worker_id: An identifier for the process or shard minting
                the ID, used to avoid collisions between concurrent
                generators. Must fit in 10 bits (``0`` to ``1023``).
                Defaults to ``0``, which is fine for a single-process
                use case.
            timestamp_ms: The Unix timestamp in milliseconds to encode.
                If ``None`` (default), the current time is used.
                Exposed mainly for deterministic testing; passing a
                value that moves the clock backward relative to the
                last call raises ``RuntimeError`` just like an actual
                backward clock movement would.

        Returns:
            A 64-bit non-negative integer, monotonically increasing
            for successive calls with the same ``worker_id`` (as long
            as the system clock does not move backward).

        Raises:
            ValueError: If ``worker_id`` does not fit in 10 bits, or if
                ``timestamp_ms`` (relative to the fixed epoch) does not
                fit in 41 bits.
            RuntimeError: If the system clock moved backward relative
                to the last call, which would otherwise risk
                generating a duplicate or decreasing ID.

        Example:
            ```pycon
            >>> from coola.identifier import SnowflakeIdGenerator
            >>> generator = SnowflakeIdGenerator()
            >>> snowflake_id = generator.generate(worker_id=3)
            >>> isinstance(snowflake_id, int)
            True

            ```
        """
        validate_bit_range(worker_id, _WORKER_ID_BITS, name="worker_id")

        with self._lock:
            if timestamp_ms is None:
                timestamp_ms = time.time_ns() // 1_000_000
            validate_bit_range(
                timestamp_ms - _EPOCH_MS, _TIMESTAMP_BITS, name="timestamp_ms - epoch"
            )
            if timestamp_ms < self._last_timestamp_ms:
                msg = (
                    f"clock moved backward: last timestamp was {self._last_timestamp_ms} ms, "
                    f"got {timestamp_ms} ms"
                )
                raise RuntimeError(msg)
            if timestamp_ms == self._last_timestamp_ms:
                self._sequence = (self._sequence + 1) & _MAX_SEQUENCE
                if self._sequence == 0:
                    # Sequence exhausted for this millisecond: spin-wait
                    # for the next one so the ID stays monotonically
                    # increasing. The lock is released for the sleep
                    # itself (re-acquired immediately after) rather than
                    # held for the whole wait, so other threads calling
                    # generate() get a chance to make progress between
                    # spins instead of being blocked until the clock
                    # ticks. Re-check for backward clock movement on
                    # every spin, since the clock could jump backward
                    # while this loop is running, which would otherwise
                    # spin forever; and re-check the shared timestamp on
                    # every spin too, in case another thread advanced it
                    # (and possibly reset the sequence) while the lock
                    # was released, so this call does not commit a
                    # timestamp/sequence pair another thread already used.
                    last_timestamp_ms = self._last_timestamp_ms
                    while timestamp_ms <= last_timestamp_ms:
                        self._lock.release()
                        try:
                            time.sleep(_SPIN_SLEEP_S)
                        finally:
                            self._lock.acquire()
                        timestamp_ms = time.time_ns() // 1_000_000
                        if timestamp_ms < last_timestamp_ms:
                            msg = (
                                f"clock moved backward: last timestamp was "
                                f"{last_timestamp_ms} ms, got {timestamp_ms} ms"
                            )
                            raise RuntimeError(msg)
                        last_timestamp_ms = max(last_timestamp_ms, self._last_timestamp_ms)
                    # timestamp_ms is now strictly greater than every
                    # last_timestamp_ms observed above (ours and any
                    # other thread's), so the sequence restarts at 0
                    # rather than keeping whatever another thread left it
                    # at for its own (earlier) timestamp.
                    self._sequence = 0
            else:
                self._sequence = 0
            self._last_timestamp_ms = timestamp_ms
            sequence = self._sequence

        return (
            ((timestamp_ms - _EPOCH_MS) << _TIMESTAMP_SHIFT)
            | (worker_id << _WORKER_ID_SHIFT)
            | sequence
        )


def get_default_generator() -> SnowflakeIdGenerator:
    r"""Get or create the default, process-wide generator backing
    ``generate_snowflake_id``.

    Returns a singleton ``SnowflakeIdGenerator`` instance shared by
    every call to ``generate_snowflake_id``. Uses the same lazy,
    singleton-on-the-function pattern as ``get_default_registry()``
    (see e.g. ``coola.equality.tester.interface``).

    Returns:
        The shared, process-wide ``SnowflakeIdGenerator`` instance.
    """
    if not hasattr(get_default_generator, "_generator"):
        get_default_generator._generator = SnowflakeIdGenerator()
    return get_default_generator._generator


def generate_snowflake_id(worker_id: int = 0, timestamp_ms: int | None = None) -> int:
    r"""Generate a Snowflake-style 64-bit identifier.

    Convenience wrapper around a shared, process-wide
    ``SnowflakeIdGenerator`` instance. Use ``SnowflakeIdGenerator``
    directly if you need multiple independent generators (e.g. one per
    worker) or want to avoid sharing state through a module-level
    singleton.

    Args:
        worker_id: An identifier for the process or shard minting the
            ID, used to avoid collisions between concurrent generators.
            Must fit in 10 bits (``0`` to ``1023``). Defaults to ``0``,
            which is fine for a single-process use case.
        timestamp_ms: The Unix timestamp in milliseconds to encode. If
            ``None`` (default), the current time is used. Exposed
            mainly for deterministic testing.

    Returns:
        A 64-bit non-negative integer, monotonically increasing for
        successive calls with the same ``worker_id`` (as long as the
        system clock does not move backward).

    Raises:
        ValueError: If ``worker_id`` does not fit in 10 bits.
        RuntimeError: If the system clock moved backward relative to
            the last call, which would otherwise risk generating a
            duplicate or decreasing ID.

    Example:
        ```pycon
        >>> from coola.identifier import generate_snowflake_id
        >>> snowflake_id = generate_snowflake_id()
        >>> isinstance(snowflake_id, int)
        True

        ```
    """
    return get_default_generator().generate(worker_id=worker_id, timestamp_ms=timestamp_ms)


def extract_snowflake_timestamp_ms(snowflake_id: int) -> int:
    r"""Extract the millisecond timestamp encoded in a Snowflake-style
    identifier.

    Inverse of the encoding done by ``SnowflakeIdGenerator.generate``
    (and ``generate_snowflake_id``): shifts out the worker ID and
    sequence fields and adds back the fixed epoch that was subtracted
    when the ID was minted.

    Args:
        snowflake_id: The identifier previously returned by
            ``SnowflakeIdGenerator.generate`` or
            ``generate_snowflake_id``.

    Returns:
        The Unix timestamp, in milliseconds, that was encoded in
        ``snowflake_id``.

    Raises:
        ValueError: If ``snowflake_id`` is negative or does not fit in
            64 bits.

    Example:
        ```pycon
        >>> from coola.identifier import extract_snowflake_timestamp_ms, generate_snowflake_id
        >>> snowflake_id = generate_snowflake_id()
        >>> isinstance(extract_snowflake_timestamp_ms(snowflake_id), int)
        True

        ```
    """
    validate_bit_range(
        snowflake_id, _TIMESTAMP_BITS + _WORKER_ID_BITS + _SEQUENCE_BITS, name="snowflake_id"
    )
    return (snowflake_id >> _TIMESTAMP_SHIFT) + _EPOCH_MS

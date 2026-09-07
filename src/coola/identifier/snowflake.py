r"""Provide a Snowflake-style identifier for time-ordered records.

Like ``generate_ulid``, a Snowflake ID is not derived from data
content: two calls produce different values even for identical input.
It packs a millisecond timestamp, a fixed worker ID, and a per-worker
sequence number into a single 64-bit integer, in the spirit of
Twitter's original Snowflake service. Prefer it over ``generate_ulid``
when the identifier must be a single 64-bit integer (e.g. a database
``BIGINT`` primary key) rather than a 26-character string, or when IDs
must be attributable to the worker/shard that minted them.
"""

from __future__ import annotations

__all__ = ["generate_snowflake_id"]

import threading
import time

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

_lock = threading.Lock()
_last_timestamp_ms = -1
_sequence = 0


def generate_snowflake_id(worker_id: int = 0) -> int:
    r"""Generate a Snowflake-style 64-bit identifier.

    The returned integer is composed of a 41-bit millisecond timestamp
    (relative to a fixed epoch), a 10-bit ``worker_id``, and a 12-bit
    sequence number that increments for IDs minted within the same
    millisecond by the same process. Because the timestamp is the most
    significant part, IDs generated later are numerically greater than
    IDs generated earlier (from the same, or an earlier, millisecond).

    Note:
        The sequence counter is process-local: it guarantees
        uniqueness for calls made from this process for a given
        ``worker_id``, not across processes. Assign each concurrently
        running process (or shard) a distinct ``worker_id`` to avoid
        collisions across processes.

    Args:
        worker_id: An identifier for the process or shard minting the
            ID, used to avoid collisions between concurrent generators.
            Must fit in 10 bits (``0`` to ``1023``). Defaults to ``0``,
            which is fine for a single-process use case.

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
    if not 0 <= worker_id <= _MAX_WORKER_ID:
        msg = (
            f"worker_id must fit in {_WORKER_ID_BITS} bits (0 to {_MAX_WORKER_ID}), got {worker_id}"
        )
        raise ValueError(msg)

    global _last_timestamp_ms, _sequence  # noqa: PLW0603
    with _lock:
        timestamp_ms = time.time_ns() // 1_000_000
        if timestamp_ms < _last_timestamp_ms:
            msg = (
                f"clock moved backward: last timestamp was {_last_timestamp_ms} ms, "
                f"got {timestamp_ms} ms"
            )
            raise RuntimeError(msg)
        if timestamp_ms == _last_timestamp_ms:
            _sequence = (_sequence + 1) & _MAX_SEQUENCE
            if _sequence == 0:
                # Sequence exhausted for this millisecond: busy-wait for
                # the next one so the ID stays monotonically increasing.
                while timestamp_ms <= _last_timestamp_ms:
                    timestamp_ms = time.time_ns() // 1_000_000
        else:
            _sequence = 0
        _last_timestamp_ms = timestamp_ms

    return (
        ((timestamp_ms - _EPOCH_MS) << _TIMESTAMP_SHIFT)
        | (worker_id << _WORKER_ID_SHIFT)
        | _sequence
    )

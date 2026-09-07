r"""Provide a MongoDB ObjectId style identifier.

Like ``SnowflakeIdGenerator``, this packs a timestamp, a fixed per-
process value, and a per-process counter into a compact identifier, so
that IDs minted later sort after IDs minted earlier. It differs from
``SnowflakeIdGenerator`` in the failure mode when the counter wraps
within a second: instead of raising or busy-waiting, the counter simply
wraps around, trading strict monotonicity guarantees for a generator
that never blocks or raises. It also does not require assigning a
distinct ``worker_id`` per process: the per-process random value below
plays that role automatically, making it convenient when there is no
existing worker/shard numbering scheme to reuse. Prefer
``SnowflakeIdGenerator`` when a numeric (rather than hex string) ID or
explicit ``worker_id`` control is needed instead.
"""

from __future__ import annotations

__all__ = ["generate_object_id"]

import os
import threading
import time

_TIMESTAMP_BYTES = 4
_PROCESS_BYTES = 5
_COUNTER_BYTES = 3
_MAX_COUNTER = (1 << (_COUNTER_BYTES * 8)) - 1

# A random 5-byte value fixed for the lifetime of this process, playing
# the role a machine/process identifier plays in the original MongoDB
# ObjectId format, without requiring the caller to configure one.
_PROCESS_VALUE = os.urandom(_PROCESS_BYTES)

_lock = threading.Lock()
_counter = int.from_bytes(os.urandom(_COUNTER_BYTES), byteorder="big")


def generate_object_id() -> str:
    r"""Generate a MongoDB ObjectId style 12-byte identifier.

    The returned value packs a 4-byte Unix timestamp (seconds), a
    5-byte value fixed once per process, and a 3-byte counter that
    increments (and silently wraps modulo ``2**24``) on every call,
    into a 24-character lowercase hex string. Because the timestamp is
    the most significant part, IDs generated in a later second sort
    (as plain strings) after IDs generated in an earlier one.

    Note:
        Unlike ``SnowflakeIdGenerator.generate``, this never raises or
        blocks: if more than ``2**24`` IDs are requested within the
        same second, the counter wraps around silently, at the cost of
        no longer guaranteeing strict ordering (or, in the extreme,
        uniqueness) for IDs minted within that second.

    Returns:
        A 24-character lowercase hex string.

    Example:
        ```pycon
        >>> from coola.identifier import generate_object_id
        >>> object_id = generate_object_id()
        >>> len(object_id)
        24

        ```
    """
    global _counter  # noqa: PLW0603
    with _lock:
        _counter = (_counter + 1) & _MAX_COUNTER
        counter = _counter
    timestamp = int(time.time()).to_bytes(_TIMESTAMP_BYTES, byteorder="big")
    payload = timestamp + _PROCESS_VALUE + counter.to_bytes(_COUNTER_BYTES, byteorder="big")
    return payload.hex()

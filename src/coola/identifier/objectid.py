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

__all__ = ["ObjectIdGenerator", "extract_object_id_timestamp", "generate_object_id"]

import os
import threading
import time

_TIMESTAMP_BYTES = 4
_PROCESS_BYTES = 5
_COUNTER_BYTES = 3
_MAX_COUNTER = (1 << (_COUNTER_BYTES * 8)) - 1


class ObjectIdGenerator:
    r"""Generate MongoDB ObjectId style 12-byte identifiers.

    All the state needed to mint IDs (the per-instance random value
    and the counter) lives on the instance rather than at module
    scope, so each generator is independent: create one per
    process/test instead of sharing mutable global state. Use
    ``generate_object_id`` for the common case of a single, shared,
    process-wide generator.

    Example:
        ```pycon
        >>> from coola.identifier.objectid import ObjectIdGenerator
        >>> generator = ObjectIdGenerator()
        >>> object_id = generator.generate()
        >>> len(object_id)
        24

        ```
    """

    def __init__(self) -> None:
        # A random 5-byte value fixed for the lifetime of this
        # generator, playing the role a machine/process identifier
        # plays in the original MongoDB ObjectId format, without
        # requiring the caller to configure one.
        self._process_value = os.urandom(_PROCESS_BYTES)
        self._lock = threading.Lock()
        self._counter = int.from_bytes(os.urandom(_COUNTER_BYTES), byteorder="big")

    def generate(self) -> str:
        r"""Generate a MongoDB ObjectId style 12-byte identifier.

        The returned value packs a 4-byte Unix timestamp (seconds), a
        5-byte value fixed once per generator instance, and a 3-byte
        counter that increments (and silently wraps modulo ``2**24``)
        on every call, into a 24-character lowercase hex string.
        Because the timestamp is the most significant part, IDs
        generated in a later second sort (as plain strings) after IDs
        generated in an earlier one.

        Note:
            Unlike ``SnowflakeIdGenerator.generate``, this never
            raises or blocks: if more than ``2**24`` IDs are requested
            within the same second, the counter wraps around silently,
            at the cost of no longer guaranteeing strict ordering (or,
            in the extreme, uniqueness) for IDs minted within that
            second.

        Returns:
            A 24-character lowercase hex string.

        Example:
            ```pycon
            >>> from coola.identifier.objectid import ObjectIdGenerator
            >>> generator = ObjectIdGenerator()
            >>> object_id = generator.generate()
            >>> len(object_id)
            24

            ```
        """
        with self._lock:
            self._counter = (self._counter + 1) & _MAX_COUNTER
            counter = self._counter
        timestamp = int(time.time()).to_bytes(_TIMESTAMP_BYTES, byteorder="big")
        payload = (
            timestamp + self._process_value + counter.to_bytes(_COUNTER_BYTES, byteorder="big")
        )
        return payload.hex()


# Default, process-wide generator backing the module-level
# `generate_object_id` function below.
_default_generator = ObjectIdGenerator()


def generate_object_id() -> str:
    r"""Generate a MongoDB ObjectId style 12-byte identifier.

    Convenience wrapper around a shared, process-wide
    ``ObjectIdGenerator`` instance. Use ``ObjectIdGenerator`` directly
    if you need multiple independent generators (e.g. one per test) or
    want to avoid sharing state through a module-level singleton.

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
    return _default_generator.generate()


def extract_object_id_timestamp(object_id: str) -> int:
    r"""Extract the Unix timestamp encoded in an ObjectId-style
    identifier.

    Inverse of the encoding done by ``ObjectIdGenerator.generate``
    (and ``generate_object_id``): decodes the leading 4 bytes of the
    24-character hex string back to the Unix timestamp (in seconds)
    it was created from.

    Args:
        object_id: The identifier string previously returned by
            ``ObjectIdGenerator.generate`` or ``generate_object_id``.

    Returns:
        The Unix timestamp, in seconds, that was encoded in
        ``object_id``.

    Raises:
        ValueError: If ``object_id`` is not a 24-character hex string.

    Example:
        ```pycon
        >>> from coola.identifier import extract_object_id_timestamp, generate_object_id
        >>> object_id = generate_object_id()
        >>> isinstance(extract_object_id_timestamp(object_id), int)
        True

        ```
    """
    if len(object_id) != 2 * (_TIMESTAMP_BYTES + _PROCESS_BYTES + _COUNTER_BYTES):
        msg = f"object_id must be a 24-character hex string, got {object_id!r}"
        raise ValueError(msg)
    try:
        payload = bytes.fromhex(object_id)
    except ValueError as error:
        msg = f"object_id must be a 24-character hex string, got {object_id!r}"
        raise ValueError(msg) from error
    return int.from_bytes(payload[:_TIMESTAMP_BYTES], byteorder="big")

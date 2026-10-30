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

__all__ = ["SnowflakeIdGenerator", "extract_snowflake_timestamp_ms"]

import secrets

from coola.identifier.sequence import TickSequence
from coola.utils.fork import track_fork
from coola.validation import validate_int, validate_range

# Custom epoch (2024-01-01T00:00:00Z, in ms since the Unix epoch) so the
# 41-bit timestamp field does not waste bits on years before this
# library existed. Shifts the field's effective range to ~69 years from
# this date.
_EPOCH_MS = 1_704_067_200_000

_TIMESTAMP_BITS = 41
_WORKER_ID_BITS = 10
_SEQUENCE_BITS = 12

_MAX_TIMESTAMP_MS = _EPOCH_MS + (1 << _TIMESTAMP_BITS) - 1
_MAX_WORKER_ID = (1 << _WORKER_ID_BITS) - 1
_MAX_SEQUENCE = (1 << _SEQUENCE_BITS) - 1

_MAX_SNOWFLAKE_ID = (1 << (_TIMESTAMP_BITS + _WORKER_ID_BITS + _SEQUENCE_BITS)) - 1

_WORKER_ID_SHIFT = _SEQUENCE_BITS
_TIMESTAMP_SHIFT = _SEQUENCE_BITS + _WORKER_ID_BITS


class SnowflakeIdGenerator:
    r"""Generate Snowflake-style 64-bit identifiers.

    An ID packs a 41-bit millisecond timestamp (counted from
    2024-01-01), the 10-bit ``worker_id`` of the generator and a 12-bit
    sequence number into an integer of at most 63 bits. IDs minted with
    the current time are strictly increasing within a generator, from
    any number of threads. The generator never raises or sleeps because
    of the clock: when more than 4,096 IDs are requested in the same
    millisecond, or when the clock moves backward, the timestamp is kept
    (or advanced by one millisecond) and the sequence goes on, so the
    timestamp of the IDs may run ahead of the clock until the clock
    catches up.

    Create one generator per process (or worker) and share it: the state
    is per generator, and there is no module-level generator because the
    ``worker_id`` can only be chosen by the caller.

    Warning:
        The uniqueness of the IDs across processes comes from the
        ``worker_id``, not from random values. Two generators that use
        the same ``worker_id`` can mint the same ID in the same
        millisecond. Assigning the worker numbers is the job of the
        caller (a database row, an environment variable, ...).

    Note:
        A generator cannot be used in the child process of a fork
        (``os.fork``, or ``multiprocessing`` with the ``fork`` start
        method): the child would continue the sequence of its parent
        with the same ``worker_id``. ``generate`` raises a
        ``RuntimeError`` in the child, which has to create a generator
        with its own ``worker_id``.

    Args:
        worker_id: The worker ID, in ``[0, 1023]``, written in every ID.

    Raises:
        TypeError: If ``worker_id`` is not an ``int`` (a ``bool`` is
            rejected).
        ValueError: If ``worker_id`` does not fit in 10 bits.

    Example:
        ```pycon
        >>> from coola.identifier import SnowflakeIdGenerator
        >>> generator = SnowflakeIdGenerator(worker_id=3)
        >>> snowflake_id = generator.generate()
        >>> isinstance(snowflake_id, int)
        True

        ```
    """

    def __init__(self, *, worker_id: int) -> None:
        validate_int(worker_id, name="worker_id")
        validate_range(worker_id, 0, _MAX_WORKER_ID, name="worker_id")
        self._worker_id = worker_id
        self._sequence = TickSequence(bits=_SEQUENCE_BITS, random_start=False)
        self._forked = False
        track_fork(self)

    def _reset_after_fork(self) -> None:
        r"""Mark the generator as unusable in the child process of a
        fork.

        The child would continue the sequence of its parent with the
        same worker ID, so ``generate`` raises an error.
        """
        # The child would continue the sequence of its parent with the same worker id.
        self._forked = True

    def generate(self, *, timestamp_ms: int | None = None) -> int:
        r"""Generate a Snowflake-style 64-bit identifier.

        Args:
            timestamp_ms: The Unix timestamp in milliseconds to encode.
                If ``None`` (default), the current time is used and the
                IDs are strictly increasing. An explicit timestamp is
                for tests and backfills: it does not touch the state of
                the generator and the sequence part is random, so such
                IDs are neither ordered nor guaranteed to be unique.

        Returns:
            A non-negative integer of at most 63 bits.

        Raises:
            RuntimeError: If the generator is used in the child
                process of a fork.
            TypeError: If ``timestamp_ms`` is not an ``int`` (a
                ``bool`` is rejected).
            ValueError: If ``timestamp_ms`` is before 2024-01-01 or
                after the last millisecond of the 41-bit field.

        Example:
            ```pycon
            >>> from coola.identifier import SnowflakeIdGenerator
            >>> generator = SnowflakeIdGenerator(worker_id=3)
            >>> snowflake_id = generator.generate()
            >>> isinstance(snowflake_id, int)
            True

            ```
        """
        if self._forked:
            msg = (
                "SnowflakeIdGenerator cannot be used in a forked process: "
                "create one with its own worker_id"
            )
            raise RuntimeError(msg)
        if timestamp_ms is None:
            timestamp_ms, sequence = self._sequence.next()
            if not _EPOCH_MS <= timestamp_ms <= _MAX_TIMESTAMP_MS:
                validate_range(timestamp_ms, _EPOCH_MS, _MAX_TIMESTAMP_MS, name="timestamp_ms")
        else:
            validate_int(timestamp_ms, name="timestamp_ms")
            validate_range(timestamp_ms, _EPOCH_MS, _MAX_TIMESTAMP_MS, name="timestamp_ms")
            sequence = secrets.randbits(_SEQUENCE_BITS)
        return (
            ((timestamp_ms - _EPOCH_MS) << _TIMESTAMP_SHIFT)
            | (self._worker_id << _WORKER_ID_SHIFT)
            | sequence
        )


def extract_snowflake_timestamp_ms(snowflake_id: int) -> int:
    r"""Extract the millisecond timestamp encoded in a Snowflake-style
    identifier.

    Inverse of the encoding done by ``SnowflakeIdGenerator.generate``
    : shifts out the worker ID and
    sequence fields and adds back the fixed epoch that was subtracted
    when the ID was minted.

    Args:
        snowflake_id: The identifier previously returned by
            ``SnowflakeIdGenerator.generate``.

    Returns:
        The Unix timestamp, in milliseconds, that was encoded in
        ``snowflake_id``.

    Raises:
        TypeError: If ``snowflake_id`` is not an ``int`` (a ``bool`` is
            rejected).
        ValueError: If ``snowflake_id`` is negative or does not fit in
            63 bits.

    Example:
        ```pycon
        >>> from coola.identifier import SnowflakeIdGenerator, extract_snowflake_timestamp_ms
        >>> snowflake_id = SnowflakeIdGenerator(worker_id=3).generate(timestamp_ms=1704067200000)
        >>> extract_snowflake_timestamp_ms(snowflake_id)
        1704067200000

        ```
    """
    # The validators cost more than the shift: they only run for an input that is not a plain
    # ``int`` of 63 bits, to raise the error.
    if not (type(snowflake_id) is int and 0 <= snowflake_id <= _MAX_SNOWFLAKE_ID):
        validate_int(snowflake_id, name="snowflake_id")
        validate_range(snowflake_id, 0, _MAX_SNOWFLAKE_ID, name="snowflake_id")
    return (snowflake_id >> _TIMESTAMP_SHIFT) + _EPOCH_MS

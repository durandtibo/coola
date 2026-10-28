r"""Provide a MongoDB ObjectId style identifier.

Like ``generate_ulid``, an ObjectId packs a timestamp followed by random
bits into a compact identifier, so that identifiers minted later sort
after identifiers minted earlier. The timestamp has a resolution of one
second, and the 24-character lowercase hex string is not a UUID or a
Base32 string. It never blocks or raises, whatever the clock does: use
it when a hex string of 12 bytes that sorts by creation time is needed.
"""

from __future__ import annotations

__all__ = ["extract_object_id_timestamp_s", "generate_object_id"]

import re
import secrets

from coola.identifier.sequence import TickSequence
from coola.validation import validate_closed_range, validate_int, validate_str

_TIMESTAMP_BITS = 32
_MAX_TIMESTAMP_S = (1 << _TIMESTAMP_BITS) - 1
_RANDOM_BITS = 64
_LENGTH = 24
_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{24}")

# State of the ObjectIds generated with the current time: they increase within a second.
_SEQUENCE = TickSequence(bits=_RANDOM_BITS, tick_ns=1_000_000_000)


def generate_object_id(*, timestamp_s: int | None = None) -> str:
    r"""Generate a MongoDB ObjectId style 12-byte identifier.

    An ObjectId packs a 32-bit timestamp in seconds followed by 64 random
    bits into a 24-character lowercase hex string. Because the
    timestamp is the most significant part, ObjectIds generated later
    sort (as plain strings) after ObjectIds generated earlier.

    Note:
        When the current time is used (``timestamp_s=None``), ObjectIds
        are strictly increasing within a process: ObjectIds generated
        in the same second (or after the clock moved backward) keep the
        timestamp and increment the random part by one. The state is
        per process, protected by a lock, and reset in the child
        process of a fork. Across processes, ObjectIds generated in the
        same second are not ordered. When ``timestamp_s`` is given, the
        random part is always random and the state is not touched: two
        ObjectIds with the same timestamp are not ordered.

    Args:
        timestamp_s: The Unix timestamp in seconds to encode. If
            ``None`` (default), the current time is used. Exposed
            mainly for deterministic testing and for backfilling.

    Returns:
        A 24-character lowercase hex string.

    Raises:
        TypeError: If ``timestamp_s`` is not an ``int`` (a ``bool`` is
            rejected).
        ValueError: If ``timestamp_s`` does not fit in 32 bits (i.e. is
            negative or exceeds ``2**32 - 1``).

    Example:
        ```pycon
        >>> from coola.identifier import generate_object_id
        >>> object_id = generate_object_id()
        >>> len(object_id)
        24

        ```
    """
    if timestamp_s is None:
        timestamp_s, randomness = _SEQUENCE.next()
        if timestamp_s > _MAX_TIMESTAMP_S:  # the tick of the sequence is never negative
            validate_closed_range(timestamp_s, 0, _MAX_TIMESTAMP_S, name="timestamp_s")
    else:
        validate_int(timestamp_s, name="timestamp_s")
        validate_closed_range(timestamp_s, 0, _MAX_TIMESTAMP_S, name="timestamp_s")
        randomness = secrets.randbits(_RANDOM_BITS)
    return f"{(timestamp_s << _RANDOM_BITS) | randomness:0{_LENGTH}x}"


def extract_object_id_timestamp_s(object_id: str) -> int:
    r"""Extract the Unix timestamp encoded in an ObjectId-style
    identifier.

    Inverse of the encoding done by ``generate_object_id``: decodes the
    leading 4 bytes of the 24-character hex string back to the Unix
    timestamp (in seconds) it was created from.

    Args:
        object_id: The identifier string previously returned by
            ``generate_object_id``.

    Returns:
        The Unix timestamp, in seconds, that was encoded in
        ``object_id``.

    Raises:
        TypeError: If ``object_id`` is not a ``str``.
        ValueError: If ``object_id`` is not a 24-character lowercase
            hex string (no whitespace, no upper case, no ``bytes``).

    Example:
        ```pycon
        >>> from coola.identifier import extract_object_id_timestamp_s, generate_object_id
        >>> extract_object_id_timestamp_s(generate_object_id(timestamp_s=1704067200))
        1704067200

        ```
    """
    validate_str(object_id, name="object_id")
    if _OBJECT_ID_PATTERN.fullmatch(object_id) is None:
        msg = f"object_id must be a {_LENGTH}-character lowercase hex string, got {object_id!r}"
        raise ValueError(msg)
    return int(object_id[: _TIMESTAMP_BITS // 4], 16)

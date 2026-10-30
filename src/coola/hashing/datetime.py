r"""Define the datetime hasher."""

from __future__ import annotations

__all__ = ["DatetimeHasher"]

from contextlib import suppress
from datetime import UTC, date, datetime, time, timedelta

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, tagged


class DatetimeHasher(InlineDisplayMixin, BaseHasher[date | time]):
    r"""Hasher for ``datetime.date``, ``datetime.datetime``
    and ``datetime.time`` objects.

    This hasher converts the object to its ISO 8601 string representation
    via ``isoformat()`` and hashes that string under a tag made of the
    concrete class name.

    Since ``datetime.datetime`` is a subclass of ``datetime.date``,
    both types are handled by this hasher. The class name is part of the
    tag, so a ``date`` and a ``datetime`` never produce the same hash,
    even for the same calendar day.

    Aware datetimes are converted to UTC first, so equal instants in
    different time zones hash the same, like they compare equal. Naive
    and aware datetimes differ. An aware ``time`` is normalised the same
    way, like ``time.__eq__`` does: the offset is subtracted from the time
    of day (without wrapping around midnight) and the payload is the
    resulting number of microseconds. A naive ``time`` uses its ISO string.

    Example:
        ```pycon
        >>> from datetime import date
        >>> from coola.hashing import DatetimeHasher
        >>> hasher = DatetimeHasher()
        >>> hasher
        DatetimeHasher()
        >>> hasher.encode(date(2021, 1, 1))
        b'date\x002021-01-01'

        ```
    """

    def encode(self, data: date | time) -> Encoding:
        r"""Describe a date, a datetime or a time.

        Args:
            data: The value to describe.

        Returns:
            The leaf ``tagged(tag, payload)``, with the tag ``date``,
            ``datetime`` or ``time`` and the ISO form as payload (the
            microseconds in UTC for an aware time).
        """
        if isinstance(data, datetime) and data.utcoffset() is not None:
            # Aware datetimes that represent the same instant are equal.
            # Near ``datetime.min``/``datetime.max`` the UTC instant may not
            # be representable, so the value is hashed with its own offset.
            with suppress(OverflowError):
                data = data.astimezone(UTC)
        if isinstance(data, time):
            offset = data.utcoffset()
            if offset is not None:
                # Aware times that compare equal have the same time of day in UTC.
                elapsed = timedelta(
                    hours=data.hour,
                    minutes=data.minute,
                    seconds=data.second,
                    microseconds=data.microsecond,
                )
                return tagged(_tag(data), f"utc:{(elapsed - offset) // timedelta(microseconds=1)}")
        return tagged(_tag(data), data.isoformat())


def _tag(data: date | time) -> str:
    r"""Return the constant tag of the family of ``data``, so a subclass
    (such as ``pandas.Timestamp``) hashes like its base.

    Args:
        data: The date, time or datetime.

    Returns:
        ``"datetime"``, ``"time"`` or ``"date"``.
    """
    if isinstance(data, datetime):
        return "datetime"
    if isinstance(data, time):
        return "time"
    return "date"

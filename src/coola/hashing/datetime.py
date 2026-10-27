r"""Define the datetime hasher."""

from __future__ import annotations

__all__ = ["DatetimeHasher"]

from contextlib import suppress
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


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
        >>> from coola.hashing import DatetimeHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = DatetimeHasher()
        >>> hasher
        DatetimeHasher()
        >>> len(hasher.hash(date(2021, 1, 1), registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: date | time,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        r"""Compute a deterministic hash of a date/datetime.

        Args:
            data: The date or datetime to hash.
            registry: The hasher registry. Unused by this hasher since
                the ISO string is hashed directly with no need to
                dispatch to another hasher for nested data; accepted
                only to satisfy the common ``BaseHasher`` interface.
            length: The desired length of the returned hex string. See
                ``hash_string`` for constraints. Defaults to 64.
            ignore_unhashable: Unused by this hasher; accepted only to
                satisfy the common ``BaseHasher`` interface.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``length`` is not an even number between 2
                and 128.
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
                return hash_tagged(
                    _tag(data),
                    f"utc:{(elapsed - offset) // timedelta(microseconds=1)}",
                    length=length,
                )
        return hash_tagged(_tag(data), data.isoformat(), length=length)


def _tag(data: date | time) -> str:
    r"""Return the constant tag of the family of ``data``, so a subclass
    (such as ``pandas.Timestamp``) hashes like its base."""
    if isinstance(data, datetime):
        return "datetime"
    if isinstance(data, time):
        return "time"
    return "date"

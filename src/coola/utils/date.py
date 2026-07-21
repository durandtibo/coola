r"""Contain utility functions to manipulate dates."""

from __future__ import annotations

__all__ = ["to_date", "to_datetime"]

from datetime import date, datetime, timezone


def to_date(d: date | str, date_format: str = "%Y-%m-%d") -> date:
    r"""Convert the input to a ``date`` object.

    Args:
        d: The date to convert. If a string is given, it is parsed
            using ``date_format``.
        date_format: A :func:`~datetime.datetime.strptime` format
            string used to parse ``d`` when it is a string.
            Defaults to ``"%Y-%m-%d"``.

    Returns:
        The input as a ``date`` object.

    Example:
        ```pycon
        >>> from datetime import date
        >>> from coola.utils.date import to_date
        >>> to_date("2026-03-19")
        datetime.date(2026, 3, 19)
        >>> to_date(date(2026, 3, 19))
        datetime.date(2026, 3, 19)

        ```
    """
    if isinstance(d, str):
        return datetime.strptime(d, date_format).replace(tzinfo=timezone.utc).date()
    return d


def to_datetime(d: datetime | str) -> datetime:
    r"""Convert the input to a ``datetime`` object.

    Args:
        d: The datetime to convert. If a string is given, it is
            parsed using :meth:`~datetime.datetime.fromisoformat`.

    Returns:
        The input as a ``datetime`` object.

    Example:
        ```pycon
        >>> from datetime import datetime
        >>> from coola.utils.date import to_datetime
        >>> to_datetime("2026-03-19T12:00:00")
        datetime.datetime(2026, 3, 19, 12, 0)
        >>> to_datetime(datetime(2026, 3, 19, 12, 0))
        datetime.datetime(2026, 3, 19, 12, 0)

        ```
    """
    if isinstance(d, str):
        return datetime.fromisoformat(d)
    return d

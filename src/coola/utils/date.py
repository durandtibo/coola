r"""Contain utility functions to manipulate dates."""

from __future__ import annotations

__all__ = ["DEFAULT_DATE_FORMAT", "get_today_date", "to_date", "to_datetime"]

from datetime import date, datetime, timezone
from typing import Final
from zoneinfo import ZoneInfo

DEFAULT_DATE_FORMAT: Final[str] = "%Y-%m-%d"


def get_today_date(timezone: str = "UTC", date_format: str = DEFAULT_DATE_FORMAT) -> str:
    r"""Return the current date for the specified timezone.

    Args:
        timezone: The IANA timezone name to use when determining
            the current date. Defaults to ``"UTC"``.
        date_format: A :func:`~datetime.datetime.strftime` format
            string used to format the returned date.
            Defaults to ``"%Y-%m-%d"``.

    Returns:
        A string representing the current date in the given timezone,
            formatted according to ``date_format``.

    Example:
        ```pycon
        >>> from coola.utils.date import get_today_date
        >>> date = get_today_date()
        >>> isinstance(date, str)
        True
        >>> date = get_today_date(timezone="America/New_York", date_format="%d/%m/%Y")
        >>> isinstance(date, str)
        True

        ```
    """
    return datetime.now(ZoneInfo(timezone)).strftime(date_format)


def to_date(d: date | str, date_format: str = DEFAULT_DATE_FORMAT) -> date:
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
        # tzinfo is only set to satisfy ruff's DTZ007; it is discarded by ``.date()``.
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

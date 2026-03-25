r"""Contain utility functions for today."""

from __future__ import annotations

__all__ = ["get_today_date"]

from datetime import datetime
from zoneinfo import ZoneInfo


def get_today_date(timezone: str = "UTC", date_format: str = "%Y-%m-%d") -> str:
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
        >>> from coola.utils.today import get_today_date
        >>> date = get_today_date()
        >>> isinstance(date, str)
        True
        >>> date = get_today_date(timezone="America/New_York", date_format="%d/%m/%Y")
        >>> isinstance(date, str)
        True

        ```
    """
    return datetime.now(ZoneInfo(timezone)).strftime(date_format)

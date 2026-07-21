from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from coola.utils.date import to_date, to_datetime

############################
#     Tests for to_date    #
############################


def test_to_date_from_date() -> None:
    """Test that a date object is returned unchanged."""
    d = date(2026, 3, 19)
    assert to_date(d) is d


def test_to_date_from_str_default_format() -> None:
    """Test that a string is parsed using the default format."""
    assert to_date("2026-03-19") == date(2026, 3, 19)


def test_to_date_from_str_custom_format() -> None:
    """Test that a string is parsed using a custom format."""
    assert to_date("19/03/2026", date_format="%d/%m/%Y") == date(2026, 3, 19)


def test_to_date_from_str_invalid_format() -> None:
    """Test that an invalid string raises the built-in exception."""
    with pytest.raises(ValueError, match="time data"):
        to_date("2026-03-19", date_format="%d/%m/%Y")


################################
#     Tests for to_datetime    #
################################


def test_to_datetime_from_datetime() -> None:
    """Test that a datetime object is returned unchanged."""
    dt = datetime(2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc)
    assert to_datetime(dt) is dt


def test_to_datetime_from_str() -> None:
    """Test that an ISO format string is parsed."""
    assert to_datetime("2026-03-19T12:00:00+00:00") == datetime(
        2026, 3, 19, 12, 0, 0, tzinfo=timezone.utc
    )


def test_to_datetime_from_str_invalid_format() -> None:
    """Test that an invalid string raises the built-in exception."""
    with pytest.raises(ValueError, match="Invalid isoformat string"):
        to_datetime("2026/03/19 12:00:00")

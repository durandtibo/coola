from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from coola.utils.date import get_today_date, to_date, to_datetime

####################################
#     Tests for get_today_date     #
####################################


def test_get_today_date_default() -> None:
    """Test that it defaults to UTC and YYYY-MM-DD format."""
    with patch("coola.utils.date.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(
            year=2026, month=3, day=19, hour=12, minute=0, second=0, tzinfo=ZoneInfo("UTC")
        )
        result = get_today_date()
        assert result == "2026-03-19"
        mock_datetime.now.assert_called_once_with(ZoneInfo("UTC"))


def test_get_today_date_custom_timezone() -> None:
    """Test that it correctly applies a specific timezone."""
    with patch("coola.utils.date.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(
            year=2026, month=3, day=20, hour=3, minute=0, second=0, tzinfo=ZoneInfo("Asia/Tokyo")
        )
        result = get_today_date(timezone="Asia/Tokyo")
        assert result == "2026-03-20"
        mock_datetime.now.assert_called_once_with(ZoneInfo("Asia/Tokyo"))


def test_get_today_date_custom_format() -> None:
    """Test that it respects a custom date format string."""
    with patch("coola.utils.date.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(
            year=2026, month=3, day=19, hour=12, minute=0, second=0, tzinfo=ZoneInfo("UTC")
        )
        result = get_today_date(date_format="%d-%m-%Y")
        assert result == "19-03-2026"
        mock_datetime.now.assert_called_once_with(ZoneInfo("UTC"))


def test_get_today_date_invalid_timezone() -> None:
    """Test that an invalid timezone string raises the built-in
    exception."""
    with pytest.raises(ZoneInfoNotFoundError, match="No time zone found with key Fake/Timezone"):
        get_today_date(timezone="Fake/Timezone")


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

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest

from coola.utils.today import get_today_date

####################################
#     Tests for get_today_date     #
####################################


def test_get_today_date_default() -> None:
    """Test that it defaults to UTC and YYYY-MM-DD format."""
    with patch("coola.utils.today.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(
            year=2026, month=3, day=19, hour=12, minute=0, second=0, tzinfo=ZoneInfo("UTC")
        )
        result = get_today_date()
        assert result == "2026-03-19"
        mock_datetime.now.assert_called_once_with(ZoneInfo("UTC"))


def test_get_today_date_custom_timezone() -> None:
    """Test that it correctly applies a specific timezone."""
    with patch("coola.utils.today.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(
            year=2026, month=3, day=20, hour=3, minute=0, second=0, tzinfo=ZoneInfo("Asia/Tokyo")
        )
        result = get_today_date(timezone="Asia/Tokyo")
        assert result == "2026-03-20"
        mock_datetime.now.assert_called_once_with(ZoneInfo("Asia/Tokyo"))


def test_get_today_date_custom_format() -> None:
    """Test that it respects a custom date format string."""
    with patch("coola.utils.today.datetime") as mock_datetime:
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

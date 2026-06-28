"""Unit tests for configure_colorlog."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.display.colorlog import configure_colorlog_logging

if TYPE_CHECKING:
    from collections.abc import Generator

MODULE = "coola.display.colorlog"


@pytest.fixture(autouse=True)
def reset_root_logger() -> Generator[None]:
    """Restore the root logger's handlers and level after each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.level = original_level


##########################################
#     Tests for configure_colorlog       #
##########################################


def test_configure_colorlog_logging() -> None:
    configure_colorlog_logging(force=True)


def test_configure_colorlog_logging_without_colorlog() -> None:
    with patch(f"{MODULE}.is_colorlog_available", return_value=False):
        configure_colorlog_logging(force=True)


def test_configure_colorlog_logging_returns_none() -> None:
    assert configure_colorlog_logging(force=True) is None


def test_configure_colorlog_logging_without_colorlog_returns_none() -> None:
    with patch(f"{MODULE}.is_colorlog_available", return_value=False):
        assert configure_colorlog_logging(force=True) is None


@pytest.mark.parametrize(
    "level",
    [
        pytest.param(logging.DEBUG, id="debug"),
        pytest.param(logging.INFO, id="info"),
        pytest.param(logging.WARNING, id="warning"),
        pytest.param(logging.ERROR, id="error"),
        pytest.param(logging.CRITICAL, id="critical"),
    ],
)
def test_configure_colorlog_logging_level(level: int) -> None:
    with patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig:
        configure_colorlog_logging(level=level)
    assert mock_basicconfig.call_args.kwargs["level"] == level


@pytest.mark.parametrize(
    "level",
    [
        pytest.param(logging.DEBUG, id="debug"),
        pytest.param(logging.INFO, id="info"),
        pytest.param(logging.WARNING, id="warning"),
        pytest.param(logging.ERROR, id="error"),
        pytest.param(logging.CRITICAL, id="critical"),
    ],
)
def test_configure_colorlog_logging_level_without_colorlog(level: int) -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=False),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(level=level)
    assert mock_basicconfig.call_args.kwargs["level"] == level


def test_configure_colorlog_logging_force_forwarded() -> None:
    with patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig:
        configure_colorlog_logging(force=True)
    assert mock_basicconfig.call_args.kwargs["force"] is True


def test_configure_colorlog_logging_force_forwarded_without_colorlog() -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=False),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(force=True)
    assert mock_basicconfig.call_args.kwargs["force"] is True


def test_configure_colorlog_logging_force_false_is_noop_when_handlers_exist() -> None:
    sentinel = logging.StreamHandler()
    logging.getLogger().addHandler(sentinel)

    with patch(f"{MODULE}.is_colorlog_available", return_value=False):
        configure_colorlog_logging(force=False)

    assert sentinel in logging.getLogger().handlers

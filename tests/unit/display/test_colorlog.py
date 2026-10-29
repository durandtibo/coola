"""Unit tests for configure_colorlog."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.display.colorlog import configure_colorlog_logging
from coola.testing.marks import colorlog_available
from coola.utils.imports import is_colorlog_available

if is_colorlog_available():
    import colorlog

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


########################################
#     Tests for configure_colorlog     #
########################################


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


######################################################
#     Tests for non-interactive (non-TTY) output     #
######################################################


def _configure_without_a_tty() -> tuple[Mock, Mock]:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.sys.stderr.isatty", return_value=False),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
        patch(f"{MODULE}.colorlog.StreamHandler") as mock_stream_handler,
    ):
        configure_colorlog_logging(level=logging.DEBUG, force=True)
    return mock_basicconfig, mock_stream_handler


@colorlog_available
def test_configure_colorlog_logging_not_a_tty_does_not_create_a_colored_handler() -> None:
    _, mock_stream_handler = _configure_without_a_tty()
    mock_stream_handler.assert_not_called()


@colorlog_available
def test_configure_colorlog_logging_not_a_tty_falls_back_to_basic_config() -> None:
    mock_basicconfig, _ = _configure_without_a_tty()
    mock_basicconfig.assert_called_once_with(level=logging.DEBUG, force=True)


def _configure_with_a_tty() -> Mock:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.sys.stderr.isatty", return_value=True),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(level=logging.DEBUG, force=True)
    return mock_basicconfig


@colorlog_available
def test_configure_colorlog_logging_tty_sets_the_level() -> None:
    assert _configure_with_a_tty().call_args.kwargs["level"] == logging.DEBUG


@colorlog_available
def test_configure_colorlog_logging_tty_sets_force() -> None:
    assert _configure_with_a_tty().call_args.kwargs["force"] is True


@colorlog_available
def test_configure_colorlog_logging_tty_attaches_one_handler() -> None:
    assert len(_configure_with_a_tty().call_args.kwargs["handlers"]) == 1


@colorlog_available
def test_configure_colorlog_logging_tty_attaches_colored_handler() -> None:
    handlers = _configure_with_a_tty().call_args.kwargs["handlers"]
    assert isinstance(handlers[0], colorlog.StreamHandler)


def test_configure_colorlog_logging_no_colorlog_ignores_tty() -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=False),
        patch(f"{MODULE}.sys.stderr.isatty", return_value=True),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(level=logging.DEBUG, force=True)

    mock_basicconfig.assert_called_once_with(level=logging.DEBUG, force=True)


############################
#     Tests for kwargs     #
############################


def test_configure_colorlog_logging_kwargs_forwarded() -> None:
    with patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig:
        configure_colorlog_logging(datefmt="%H:%M")
    assert mock_basicconfig.call_args.kwargs["datefmt"] == "%H:%M"


def test_configure_colorlog_logging_kwargs_forwarded_without_colorlog() -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=False),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(force=True, datefmt="%H:%M")
    mock_basicconfig.assert_called_once_with(level=logging.INFO, force=True, datefmt="%H:%M")


@colorlog_available
def test_configure_colorlog_logging_kwargs_forwarded_tty() -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.sys.stderr.isatty", return_value=True),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(datefmt="%H:%M")
    assert mock_basicconfig.call_args.kwargs["datefmt"] == "%H:%M"


########################################
#     Tests for the tty and kwargs     #
########################################


@colorlog_available
def test_configure_colorlog_logging_tty_stream_is_used_by_handler() -> None:
    stream = io.StringIO()
    stream.isatty = lambda: True  # ty: ignore[invalid-assignment]
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(stream=stream)
    assert mock_basicconfig.call_args.kwargs["handlers"][0].stream is stream


@colorlog_available
def test_configure_colorlog_logging_tty_stream_is_not_forwarded() -> None:
    stream = io.StringIO()
    stream.isatty = lambda: True  # ty: ignore[invalid-assignment]
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(stream=stream)
    assert "stream" not in mock_basicconfig.call_args.kwargs


@colorlog_available
def test_configure_colorlog_logging_stream_not_a_tty_falls_back_to_basic_config() -> None:
    stream = io.StringIO()
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.sys.stderr.isatty", return_value=True),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(stream=stream)
    mock_basicconfig.assert_called_once_with(level=logging.INFO, force=False, stream=stream)


@colorlog_available
@pytest.mark.parametrize(
    "kwargs",
    [
        {"handlers": []},
        {"filename": "app.log"},
        {"filemode": "a"},
        {"format": "%(message)s"},
        {"style": "{"},
    ],
)
def test_configure_colorlog_logging_tty_plain_options_forwarded_without_handler(
    kwargs: dict,
) -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.sys.stderr.isatty", return_value=True),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(**kwargs)
    mock_basicconfig.assert_called_once_with(level=logging.INFO, force=False, **kwargs)


@colorlog_available
def test_configure_colorlog_logging_tty_datefmt_used_by_formatter() -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.sys.stderr.isatty", return_value=True),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging(datefmt="%H:%M")
    handler = mock_basicconfig.call_args.kwargs["handlers"][0]
    assert handler.formatter.datefmt == "%H:%M"


@colorlog_available
def test_configure_colorlog_logging_stderr_none_falls_back_to_basic_config() -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", return_value=True),
        patch(f"{MODULE}.sys.stderr", None),
        patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig,
    ):
        configure_colorlog_logging()
    mock_basicconfig.assert_called_once_with(level=logging.INFO, force=False)


def test_configure_colorlog_logging_stream_without_isatty_falls_back_to_basic_config() -> None:
    with patch(f"{MODULE}.logging.basicConfig") as mock_basicconfig:
        configure_colorlog_logging(stream=object())  # ty: ignore[invalid-argument-type]
    mock_basicconfig.assert_called_once()


def test_configure_colorlog_logging_force_positional_raises() -> None:
    with pytest.raises(TypeError, match=r"positional argument"):
        configure_colorlog_logging(logging.INFO, True)  # ty: ignore[too-many-positional-arguments]


def test_configure_colorlog_logging_handlers_and_stream_raises() -> None:
    with pytest.raises(ValueError, match=r"should not be specified together with 'handlers'"):
        configure_colorlog_logging(force=True, handlers=[], stream=io.StringIO())


def test_configure_colorlog_logging_unknown_option_raises() -> None:
    with pytest.raises(ValueError, match=r"Unrecognised argument"):
        configure_colorlog_logging(force=True, unknown=1)

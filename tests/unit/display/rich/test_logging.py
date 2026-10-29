from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.display.rich import configure_rich_logging
from coola.testing.marks import rich_available
from coola.utils.imports import is_rich_available

if TYPE_CHECKING:
    from collections.abc import Generator


if is_rich_available():
    from rich import get_console
    from rich.console import Console
    from rich.logging import RichHandler

pytestmark = rich_available

MODULE = "coola.display.rich.logging"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_root_logger() -> Generator[None]:
    """Restore the root logger's handlers and level after each test.

    pytest installs its own ``LogCaptureHandler`` before each test, so
    we snapshot and restore rather than clearing, to avoid breaking log
    capture.
    """
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.level = original_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rich_handler() -> RichHandler:
    """Return the RichHandler on the root logger, or raise if absent."""
    for h in logging.getLogger().handlers:
        if isinstance(h, RichHandler):
            return h
    msg = f"No RichHandler found on root logger. Handlers: {logging.getLogger().handlers}"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------


def test_returns_none() -> None:
    assert configure_rich_logging(force=True) is None


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_installs_rich_handler() -> None:
    configure_rich_logging(force=True)
    _rich_handler()  # raises if not found


def test_installs_exactly_one_rich_handler() -> None:
    configure_rich_logging(force=True)
    rich_handlers = [h for h in logging.getLogger().handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) == 1


def test_default_level_is_info() -> None:
    configure_rich_logging(force=True)
    assert logging.getLogger().level == logging.INFO


def test_default_rich_tracebacks_is_true() -> None:
    configure_rich_logging(force=True)
    assert _rich_handler().rich_tracebacks is True


def test_default_markup_is_true() -> None:
    configure_rich_logging(force=True)
    assert _rich_handler().markup is True


# ---------------------------------------------------------------------------
# Level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level",
    [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL],
)
def test_custom_level(level: int) -> None:
    configure_rich_logging(level=level, force=True)
    assert logging.getLogger().level == level


# ---------------------------------------------------------------------------
# RichHandler options
# ---------------------------------------------------------------------------


def test_rich_tracebacks_false() -> None:
    configure_rich_logging(rich_tracebacks=False, force=True)
    assert _rich_handler().rich_tracebacks is False


def test_markup_false() -> None:
    configure_rich_logging(markup=False, force=True)
    assert _rich_handler().markup is False


def test_extra_kwargs_forwarded_to_rich_handler() -> None:
    """Keyword arguments unknown to configure_rich_logging reach
    RichHandler.

    ``show_path`` is stored internally by RichHandler and not exposed as
    a public attribute, so we verify forwarding by inspecting the
    constructor call via a mock rather than reading the attribute after
    the fact.
    """
    with patch(f"{MODULE}.RichHandler", wraps=RichHandler) as mock_handler:
        configure_rich_logging(show_path=False, force=True)
    _, kwargs = mock_handler.call_args
    assert kwargs.get("show_path") is False


# ---------------------------------------------------------------------------
# console
# ---------------------------------------------------------------------------


def test_console_is_used_by_rich_handler() -> None:
    console = Console()
    configure_rich_logging(console=console, force=True)
    assert _rich_handler().console is console


def test_default_console_is_shared_console() -> None:
    configure_rich_logging(force=True)
    assert _rich_handler().console is get_console()


# ---------------------------------------------------------------------------
# fmt / datefmt
# ---------------------------------------------------------------------------


def test_custom_fmt_is_forwarded() -> None:
    custom_fmt = "%(levelname)s %(name)s %(message)s"
    with patch("logging.basicConfig") as mock_basic_config:
        configure_rich_logging(fmt=custom_fmt)
    _, kwargs = mock_basic_config.call_args
    assert kwargs["format"] == custom_fmt


def test_custom_datefmt_is_forwarded() -> None:
    custom_datefmt = "%H:%M:%S"
    with patch("logging.basicConfig") as mock_basic_config:
        configure_rich_logging(datefmt=custom_datefmt)
    _, kwargs = mock_basic_config.call_args
    assert kwargs["datefmt"] == custom_datefmt


# ---------------------------------------------------------------------------
# force flag
# ---------------------------------------------------------------------------


def test_force_false_is_noop_when_handlers_exist() -> None:
    """BasicConfig without force=True is a no-op if handlers are already
    present."""
    sentinel = logging.StreamHandler()
    logging.getLogger().addHandler(sentinel)

    configure_rich_logging(force=False)

    assert not any(isinstance(h, RichHandler) for h in logging.getLogger().handlers)


def test_force_true_replaces_existing_handlers() -> None:
    """Force=True clears pre-existing handlers and installs
    RichHandler."""
    sentinel = logging.StreamHandler()
    logging.getLogger().addHandler(sentinel)

    configure_rich_logging(force=True)

    handlers = logging.getLogger().handlers
    assert sentinel not in handlers
    assert any(isinstance(h, RichHandler) for h in handlers)


def test_force_forwarded_to_basic_config() -> None:
    with patch("logging.basicConfig") as mock_basic_config:
        configure_rich_logging(force=True)
    _, kwargs = mock_basic_config.call_args
    assert kwargs["force"] is True


# ---------------------------------------------------------------------------
# Invalid kwargs
# ---------------------------------------------------------------------------


def test_unknown_kwarg_raises_type_error() -> None:
    with pytest.raises(TypeError):
        configure_rich_logging(this_does_not_exist=True)

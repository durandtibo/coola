"""Provide utilities for configuring Python's logging output."""

from __future__ import annotations

__all__ = ["configure_colorlog_logging"]

import logging
import sys
from typing import Any

from coola.utils.imports import is_colorlog_available

if is_colorlog_available():  # pragma: no cover
    import colorlog
else:  # pragma: no cover
    from coola.utils.fallback.colorlog import colorlog


# Options of ``logging.basicConfig`` that define where or how records are written. When
# one of them is given, the caller controls the output and no coloured handler is added.
_PLAIN_OPTIONS = frozenset({"handlers", "filename", "filemode", "format", "style"})


def _is_tty(stream: object) -> bool:
    r"""Indicate if a stream is attached to a terminal.

    ``sys.stderr`` can be ``None`` (for example with ``pythonw``).

    Args:
        stream: The stream to check.

    Returns:
        ``True`` if ``stream`` is attached to a terminal, otherwise
            ``False``.
    """
    isatty = getattr(stream, "isatty", None)
    return callable(isatty) and bool(isatty())


def configure_colorlog_logging(
    level: int = logging.INFO, *, force: bool = False, **kwargs: Any
) -> None:
    r"""Configure the root logger, using a coloured formatter when
    available.

    If the ``colorlog`` package is installed and ``sys.stderr`` is
    attached to a terminal, attaches a ``colorlog.StreamHandler``
    with per-level colours for both the log metadata (level, logger
    name, line number) and the message itself.  If ``colorlog`` is not
    installed, or output is not a terminal (e.g. redirected to a file
    or running in CI), falls back to plain ``logging.basicConfig``
    with no formatting, to avoid emitting raw ANSI escape codes into
    non-interactive output.

    The coloured handler writes to ``stream`` if given, and to
    ``sys.stderr`` otherwise; the terminal check applies to this
    stream. If ``handlers``, ``filename``, ``filemode``, ``format`` or
    ``style`` is given, the caller controls the output and the call is
    forwarded to ``logging.basicConfig`` without a coloured
    handler. ``datefmt`` is used by the coloured formatter.

    ``colorlog`` is optional, unlike ``rich`` for
    :func:`~coola.display.rich.configure_rich_logging`: without it the
    output has no colour, which is a correct result for a logging
    helper, so no error is raised.

    Note:
        ``logging.basicConfig`` is a **no-op** if the root logger
        already has handlers configured.  Pass ``force=True`` to remove
        existing handlers and reconfigure unconditionally.

    Args:
        level: Minimum log level for the root logger.  Accepts any
            constant from ``logging`` (e.g. ``logging.DEBUG``,
            ``logging.WARNING``).  Defaults to ``logging.INFO``.
        force: When ``True``, removes any existing handlers before
            applying the new configuration, ensuring this call always
            takes effect.  Defaults to ``False``.
        **kwargs: Additional keyword arguments passed to
            ``logging.basicConfig`` (e.g. ``stream``, ``filename``,
            ``datefmt``).

    Raises:
        TypeError: If ``kwargs`` contains an option that
            ``logging.basicConfig`` does not accept.
        ValueError: If ``kwargs`` contains incompatible options (for
            example ``handlers`` with ``stream``), or if ``level`` is
            not a valid level.

    Example:
        ```pycon
        >>> import logging
        >>> from coola.display.colorlog import configure_colorlog_logging
        >>> configure_colorlog_logging(level=logging.DEBUG, force=True)
        >>> logging.getLogger().level
        10

        ```
    """
    stream = kwargs.get("stream") or sys.stderr
    if not is_colorlog_available() or _PLAIN_OPTIONS & kwargs.keys() or not _is_tty(stream):
        logging.basicConfig(level=level, force=force, **kwargs)
        return

    kwargs.pop("stream", None)
    handler = colorlog.StreamHandler(stream)
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt=(
                "%(log_color)s(%(process)d) %(asctime)s [%(levelname)s] %(name)s:%(lineno)s%(reset)s "
                "%(message_log_color)s%(message)s"
            ),
            datefmt=kwargs.get("datefmt", "%Y-%m-%d %H:%M:%S"),
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "bold_yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            secondary_log_colors={
                "message": {
                    "DEBUG": "cyan",
                    "INFO": "reset",
                    "WARNING": "bold_yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            },
        )
    )
    logging.basicConfig(level=level, handlers=[handler], force=force, **kwargs)

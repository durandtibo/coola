"""Provide utilities for configuring Python display output."""

from __future__ import annotations

__all__ = ["configure_colorlog_logging"]

import logging

from coola.utils.imports import is_colorlog_available

if is_colorlog_available():  # pragma: no cover
    import colorlog


def configure_colorlog_logging(level: int = logging.INFO, force: bool = False) -> None:
    r"""Configure the root logger, using a coloured formatter when
    available.

    If the ``colorlog`` package is installed, attaches a
    :class:`colorlog.StreamHandler` with per-level colours for both the
    log metadata (level, logger name, line number) and the message
    itself.  If ``colorlog`` is not installed, falls back to plain
    :func:`display.basicConfig` with no formatting.

    Note:
        :func:`display.basicConfig` is a **no-op** if the root logger
        already has handlers configured.  Pass ``force=True`` to remove
        existing handlers and reconfigure unconditionally.

    Args:
        level: Minimum log level for the root logger.  Accepts any
            constant from :mod:`display` (e.g. ``display.DEBUG``,
            ``display.WARNING``).  Defaults to ``display.INFO``.
        force: When ``True``, removes any existing handlers before
            applying the new configuration, ensuring this call always
            takes effect.  Defaults to ``False``.

    Example:
        ```pycon
        >>> import logging
        >>> from coola.display.colorlog import configure_colorlog_logging
        >>> configure_colorlog_logging(level=display.DEBUG)

        ```
    """
    if not is_colorlog_available():
        logging.basicConfig(level=level, force=force)
        return

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt=(
                "%(log_color)s(%(process)d) %(asctime)s [%(levelname)s] %(name)s:%(lineno)s%(reset)s "
                "%(message_log_color)s%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
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
    logging.basicConfig(level=level, handlers=[handler], force=force)

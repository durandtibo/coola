"""Provide utilities for configuring Rich-powered logging output."""

from __future__ import annotations

__all__ = ["configure_rich_logging"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils.imports import is_rich_available

if TYPE_CHECKING or is_rich_available():  # pragma: no cover
    from rich import get_console
    from rich.logging import RichHandler
else:  # pragma: no cover
    from coola.utils.fallback.rich import RichHandler, get_console


def configure_rich_logging(
    level: int = logging.INFO,
    *,
    fmt: str = "%(message)s",
    datefmt: str = "[%Y-%m-%d %H:%M:%S]",
    rich_tracebacks: bool = True,
    markup: bool = True,
    force: bool = False,
    **kwargs: Any,
) -> None:
    """Configure the root logger to use
    :class:`rich.logging.RichHandler`.

    Sets up Python's standard :mod:`logging` module with a single
    :class:`~rich.logging.RichHandler` so that all log output is rendered
    through Rich's console, with coloured levels, timestamps, and optional
    pretty-printed tracebacks.

    After calling this function, obtain a logger for the current module with
    the standard idiom::

        logger = logging.getLogger(__name__)

    Using :func:`logging.getLogger` with ``__name__`` gives the logger a
    dotted name matching its import path (e.g. ``langchain_metrics.console``),
    which slots it correctly into the logging hierarchy and allows callers to
    tune or silence it by name.

    Note:
        :func:`logging.basicConfig` is a **no-op** if the root logger already
        has handlers configured (e.g. if another library called
        ``basicConfig`` first, or if you call this function more than once).
        Pass ``force=True`` to remove any existing handlers and reconfigure
        unconditionally.

    Args:
        level: Minimum log level for the root logger.  Accepts any constant
            from :mod:`logging` (e.g. ``logging.DEBUG``, ``logging.WARNING``).
            Defaults to ``logging.INFO``.
        fmt: Log record format string passed to :class:`logging.Formatter`.
            Because Rich renders level, timestamp, and logger name itself, the
            default ``"%(message)s"`` avoids duplicating those fields.  Change
            this if you need additional fields such as ``%(name)s`` in the
            formatted output.
        datefmt: Date/time format string for the ``%(asctime)s`` field,
            following :func:`time.strftime` conventions.  Defaults to
            ``"[%Y-%m-%d %H:%M:%S]"``.
        rich_tracebacks: When ``True``, exceptions are rendered by Rich as
            syntax-highlighted, multi-frame tracebacks instead of the standard
            Python traceback format.  Defaults to ``True``.
        markup: When ``True``, Rich markup tags (e.g. ``[bold red]``) in log
            messages are interpreted and rendered.  Set to ``False`` if log
            messages may contain literal square brackets that should not be
            treated as markup.  Defaults to ``True``.
        force: When ``True``, any existing handlers on the root logger are
            removed before applying the new configuration, ensuring this call
            always takes effect.  Equivalent to passing ``force=True`` to
            :func:`logging.basicConfig` (requires Python 3.8+).
            Defaults to ``False``.
        **kwargs: Additional keyword arguments forwarded verbatim to
            :class:`~rich.logging.RichHandler`.  For example::

                configure_rich_logging(show_path=False, tracebacks_show_locals=True)

            See the `Rich documentation
            <https://rich.readthedocs.io/en/latest/logging.html>`_ for the
            full list of supported parameters.

    Raises:
        MissingPackageError: If ``rich`` is not installed.
        TypeError: If an unrecognised keyword argument is passed that
            :class:`~rich.logging.RichHandler` does not accept.

    Example:
        ```pycon
        >>> import logging
        >>> from coola.display.rich import configure_rich_logging
        >>> configure_rich_logging()
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Rich logging is active")

        ```
    """
    # The handler is created first: a missing ``rich`` raises before the root logger is changed.
    handler = RichHandler(
        console=get_console(), rich_tracebacks=rich_tracebacks, markup=markup, **kwargs
    )
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=[handler], force=force)

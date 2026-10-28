r"""Contain utility functions to measure time."""

from __future__ import annotations

__all__ = ["TimingResult", "timeblock"]

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.formatting.quantity import str_time_human

if TYPE_CHECKING:
    from collections.abc import Generator

logger: logging.Logger = logging.getLogger(__name__)


class TimingResult(InlineDisplayMixin):
    r"""Hold the result of a timed block.

    Args:
        started_at: The time when the block started, or ``None`` if
            it has not yet started.
        finished_at: The time when the block finished, or ``None`` if
            it has not yet finished.
    """

    def __init__(self, started_at: float | None = None, finished_at: float | None = None) -> None:
        self.started_at = started_at
        self.finished_at = finished_at

    @property
    def elapsed(self) -> float | None:
        r"""The elapsed time in seconds, or ``None`` if the block has
        not yet completed."""
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    def _display_fields(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": self.elapsed,
        }


@contextmanager
def timeblock(message: str = "Total time: {time}") -> Generator[TimingResult]:
    r"""Implement a context manager to measure the execution time of a
    block of code.

    Args:
        message: The message displayed when the time is logged. It
            must contain the ``{time}`` placeholder.

    Yields:
        A ``TimingResult`` whose ``elapsed`` attribute is set when the
            block exits.

    Raises:
        ValueError: If ``message`` does not contain ``{time}``.

    Example:
        ```pycon
        >>> from coola.tools.timing import timeblock
        >>> with timeblock():
        ...     x = [1, 2, 3]
        ...
        >>> with timeblock("Training: {time}"):
        ...     y = [1, 2, 3]
        ...
        >>> with timeblock() as t:
        ...     z = [1, 2, 3]
        ...
        >>> t.elapsed

        ```
    """
    if "{time}" not in message:
        msg = f"{{time}} is missing in the message (received: {message})"
        raise ValueError(msg)
    result = TimingResult()
    result.started_at = time.perf_counter()
    try:
        yield result
    finally:
        result.finished_at = time.perf_counter()
        # ``replace`` (not ``format``) so other braces in the message are kept as is.
        logger.info(message.replace("{time}", str_time_human(result.elapsed or 0.0)))

r"""Contain utility functions to measure time."""

from __future__ import annotations

__all__ = ["TimingResult", "timeblock"]

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

from coola.utils.format import str_time_human

if TYPE_CHECKING:
    from collections.abc import Generator

logger: logging.Logger = logging.getLogger(__name__)


class TimingResult:
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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"started_at={self.started_at}, "
            f"finished_at={self.finished_at}, "
            f"elapsed={self.elapsed})"
        )


@contextmanager
def timeblock(message: str = "Total time: {time}") -> Generator[None, None, None]:
    r"""Implement a context manager to measure the execution time of a
    block of code.

    Args:
        message: The message displayed when the time is logged.

    Example:
        ```pycon
        >>> from coola.utils.timing import timeblock
        >>> with timeblock():
        ...     x = [1, 2, 3]
        ...
        >>> with timeblock("Training: {time}"):
        ...     y = [1, 2, 3]
        ...

        ```
    """
    if "{time}" not in message:
        msg = f"{{time}} is missing in the message (received: {message})"
        raise RuntimeError(msg)
    start_time = time.perf_counter()
    try:
        yield
    finally:
        logger.info(message.format(time=str_time_human(time.perf_counter() - start_time)))

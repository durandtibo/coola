r"""Contain utility functions to measure time."""

from __future__ import annotations

__all__ = ["timeblock"]

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

from coola.utils.format import str_time_human

if TYPE_CHECKING:
    from collections.abc import Generator

logger: logging.Logger = logging.getLogger(__name__)


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

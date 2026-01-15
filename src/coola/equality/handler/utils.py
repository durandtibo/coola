r"""Contain utilities for handlers."""

from __future__ import annotations

__all__ = ["check_recursion_depth", "handlers_are_equal"]

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from coola.equality.config import EqualityConfig
    from coola.equality.handler.base import BaseEqualityHandler


def handlers_are_equal(
    handler1: BaseEqualityHandler | None, handler2: BaseEqualityHandler | None
) -> bool:
    r"""Indicate whether two handlers are equal.

    Args:
        handler1: The first handler.
        handler2: The second handler.

    Returns:
        ``True`` if both handlers are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.equality.handler import SameObjectHandler, FalseHandler, handlers_are_equal
        >>> handlers_are_equal(SameObjectHandler(), SameObjectHandler())
        True
        >>> handlers_are_equal(SameObjectHandler(), FalseHandler())
        False
        >>> handlers_are_equal(None, SameObjectHandler())
        False
        >>> handlers_are_equal(SameObjectHandler(), None)
        False
        >>> handlers_are_equal(None, None)
        True

        ```
    """
    if handler1 is None:
        return handler1 == handler2
    return handler1.equal(handler2)


@contextmanager
def check_recursion_depth(config: EqualityConfig) -> Generator[None, None, None]:
    r"""Context manager to track and enforce recursion depth limits.

    This context manager increments the recursion depth counter on entry
    and decrements it on exit (even if an exception occurs). It raises
    a RecursionError if the maximum depth is exceeded.

    Args:
        config: The equality configuration containing depth settings.

    Raises:
        RecursionError: if the current depth exceeds max_depth.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.utils import check_recursion_depth
        >>> config = EqualityConfig(max_depth=5)
        >>> with check_recursion_depth(config):
        ...     print(config._current_depth)
        1

        ```
    """
    if config._current_depth >= config.max_depth:
        msg = (
            f"Maximum recursion depth ({config.max_depth}) exceeded. "
            f"Consider increasing max_depth parameter or simplifying the data structure."
        )
        raise RecursionError(msg)
    config._current_depth += 1
    try:
        yield
    finally:
        config._current_depth -= 1

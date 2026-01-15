r"""Contain utilities for handlers."""

from __future__ import annotations

__all__ = ["check_recursion_depth", "create_chain", "handlers_are_equal"]

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from coola.equality.config import EqualityConfig
    from coola.equality.handler.base import BaseEqualityHandler


def create_chain(*handlers: BaseEqualityHandler) -> BaseEqualityHandler:
    r"""Create a chain of handlers and return the first handler.

    Args:
        handlers: Handlers to chain.

    Returns:
        The first handler of the chain.

    Example:
        ```pycon
        >>> from coola.equality.handler import (
        ...     create_chain,
        ...     SameObjectHandler,
        ...     SameTypeHandler,
        ...     ObjectEqualHandler,
        ... )
        >>> handler = create_chain(SameObjectHandler(), SameTypeHandler(), ObjectEqualHandler())
        >>> print(handler.visualize_chain())
        (0): SameObjectHandler()
        (1): SameTypeHandler()
        (2): ObjectEqualHandler()

        ```
    """
    if not handlers:
        msg = "At least one handler is required to create a chain."
        raise ValueError(msg)
    handler = handlers[0]
    handler.chain_all(*handlers[1:])
    return handler


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
        >>> from coola.equality.handler import check_recursion_depth
        >>> config = EqualityConfig(max_depth=5)
        >>> with check_recursion_depth(config):
        ...     print(config._current_depth)
        ...
        1

        ```
    """
    if config.depth >= config.max_depth:
        msg = (
            f"Maximum recursion depth ({config.max_depth}) exceeded. "
            f"Consider increasing max_depth parameter or simplifying the data structure."
        )
        raise RecursionError(msg)
    config.increment_depth()
    try:
        yield
    finally:
        config.decrement_depth()

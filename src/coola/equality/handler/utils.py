r"""Contain utilities for handlers."""

from __future__ import annotations

__all__ = ["handlers_are_equal"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

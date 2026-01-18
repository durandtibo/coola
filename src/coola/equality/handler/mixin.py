r"""Define the equality handler base classes."""

from __future__ import annotations

__all__ = ["HandlerEqualityMixin"]

from typing import TYPE_CHECKING

from coola.equality.handler.utils import handlers_are_equal

if TYPE_CHECKING:
    from coola.equality.handler.base import BaseEqualityHandler


class HandlerEqualityMixin:
    r"""Mixin providing a standard implementation of the equal() method.

    This mixin eliminates code duplication across handlers that only need
    to compare their type and next_handler. Handlers using this mixin must
    inherit from BaseEqualityHandler to ensure the next_handler attribute
    is available.

    Design Note:
        This mixin must be used with classes that inherit from
        ``BaseEqualityHandler``. The type annotation ``self: BaseEqualityHandler``
        on the ``equal()`` method enforces this constraint and enables
        type-safe access to the ``next_handler`` attribute.

    Example:
        ```pycon
        >>> from coola.equality.handler import BaseEqualityHandler, HandlerEqualityMixin
        >>> class MyHandler(HandlerEqualityMixin, BaseEqualityHandler):
        ...     def handle(self, actual, expected, config):
        ...         return True
        ...
        >>> handler1 = MyHandler()
        >>> handler2 = MyHandler()
        >>> handler1.equal(handler2)
        True

        ```
    """

    def equal(self: BaseEqualityHandler, other: object) -> bool:
        r"""Indicate if two handlers are equal.

        Two handlers are equal if they are of the same type and have
        equal next_handler chains.

        Note:
            The type annotation ``self: BaseEqualityHandler`` ensures this
            mixin is only used with BaseEqualityHandler subclasses, enabling
            type-safe access to ``next_handler``.

        Args:
            other: The other object to compare with.

        Returns:
            ``True`` if the handlers are equal, otherwise ``False``.
        """
        if type(other) is not type(self):
            return False
        return handlers_are_equal(self.next_handler, other.next_handler)

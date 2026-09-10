r"""Define the equality handler base classes."""

from __future__ import annotations

__all__ = ["EqualityHandler", "HandlerEqualityMixin"]

from typing import TYPE_CHECKING, Protocol

from coola.equality.handler.utils import handlers_are_equal

if TYPE_CHECKING:
    from coola.equality.handler.base import BaseEqualityHandler


class EqualityHandler(Protocol):
    r"""Protocol describing the members ``HandlerEqualityMixin.equal()``
    relies on: those of ``BaseEqualityHandler`` (``next_handler``) plus
    ``_equality_attrs()`` from the mixin.

    This lets ``equal()`` type-check access to ``self`` without a
    runtime (import-time) dependency between ``HandlerEqualityMixin``
    and ``BaseEqualityHandler``.
    """

    @property
    def next_handler(self) -> BaseEqualityHandler | None: ...

    def _equality_attrs(self) -> tuple[str, ...]: ...


class HandlerEqualityMixin:
    r"""Mixin providing a standard implementation of the equal() method.

    This mixin eliminates code duplication across handlers that only need
    to compare their type and next_handler. Handlers using this mixin must
    inherit from BaseEqualityHandler to ensure the next_handler attribute
    is available.

    Design Note:
        This mixin must be used with classes that inherit from
        ``BaseEqualityHandler``. The type annotation ``self: EqualityHandler``
        on the ``equal()`` method enforces this constraint and enables
        type-safe access to the ``next_handler`` attribute (and, for
        handlers with extra state, ``_equality_attrs()``).

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

    A handler that also holds extra state (e.g. a configuration field
    besides ``next_handler``) can still use this mixin by overriding
    ``_equality_attrs()`` to name the extra attributes to compare:

    Example:
        ```pycon
        >>> from coola.equality.handler import BaseEqualityHandler, HandlerEqualityMixin
        >>> class MyHandler(HandlerEqualityMixin, BaseEqualityHandler):
        ...     def __init__(self, name, next_handler=None):
        ...         super().__init__(next_handler=next_handler)
        ...         self.name = name
        ...     def _equality_attrs(self):
        ...         return ("name",)
        ...     def handle(self, actual, expected, config):
        ...         return True
        ...
        >>> handler1 = MyHandler(name="shape")
        >>> handler2 = MyHandler(name="shape")
        >>> handler3 = MyHandler(name="dtype")
        >>> handler1.equal(handler2)
        True
        >>> handler1.equal(handler3)
        False

        ```
    """

    def _equality_attrs(self) -> tuple[str, ...]:
        r"""Name the extra instance attributes ``equal`` should compare.

        Override this in a subclass that has extra state beyond
        ``next_handler``. The default is an empty tuple, i.e. only the
        type and the ``next_handler`` chain are compared.

        Returns:
            The names of the extra attributes to compare.
        """
        return ()

    def equal(self: EqualityHandler, other: object) -> bool:
        r"""Indicate if two handlers are equal.

        Two handlers are equal if they are of the same type, have equal
        values for the attributes named by ``_equality_attrs()``, and
        have equal ``next_handler`` chains.

        Note:
            The type annotation ``self: EqualityHandler`` ensures this
            mixin is only used with BaseEqualityHandler subclasses, enabling
            type-safe access to ``next_handler`` and ``_equality_attrs``.

        Args:
            other: The other object to compare with.

        Returns:
            ``True`` if the handlers are equal, otherwise ``False``.
        """
        if type(other) is not type(self):
            return False
        for attr in self._equality_attrs():
            if getattr(self, attr) != getattr(other, attr):
                return False
        return handlers_are_equal(self.next_handler, other.next_handler)

r"""Define the equality handler base classes."""

from __future__ import annotations

__all__ = ["BaseEqualityHandler"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

T = TypeVar("T", bound="BaseEqualityHandler")


class BaseEqualityHandler(ABC):
    r"""Define the base class to implement an equality handler.

    A child class needs to implement the following methods:

    - ``handle``

    A terminal handler has its next handler set to ``None``.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import SameObjectHandler, FalseHandler
        >>> config = EqualityConfig()
        >>> handler = SameObjectHandler()
        >>> handler.set_next_handler(FalseHandler())
        >>> handler.handle("abc", "abc", config)
        True
        >>> handler.handle("abc", "ABC", config)
        False

        ```
    """

    def __init__(self, next_handler: BaseEqualityHandler | None = None) -> None:
        self._verify_next_handler(next_handler)
        self._next_handler = next_handler

    @property
    def next_handler(self) -> BaseEqualityHandler | None:
        r"""The next handler."""
        return self._next_handler

    @abstractmethod
    def equal(self, other: object) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            other: The other object.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example:
            ```pycon
            >>> from coola.equality.handler import SameObjectHandler, TrueHandler
            >>> handler1 = SameObjectHandler()
            >>> handler2 = SameObjectHandler()
            >>> handler3 = TrueHandler()
            >>> handler1.equal(handler2)
            True
            >>> handler1.equal(handler3)
            False

            ```
        """

    @abstractmethod
    def handle(self, actual: object, expected: object, config: EqualityConfig) -> bool:
        r"""Return the equality result between the two input objects.

        Args:
            actual: The actual input.
            expected: The expected input.
            config: The equality configuration.

        Returns:
            ``True`` if the input objects are equal, and ``False``
                otherwise.

        Example:
            ```pycon
            >>> from coola.equality.config import EqualityConfig
            >>> from coola.equality.handler import SameObjectHandler
            >>> config = EqualityConfig()
            >>> handler = SameObjectHandler()
            >>> handler.handle("abc", "abc", config)
            True

            ```
        """

    def chain(self, handler: T) -> T:
        r"""Chain a handler to the current handler.

        Args:
            handler: The handler to chain.

        Returns:
            The input handler.

        Example:
            ```pycon
            >>> from coola.equality.config import EqualityConfig
            >>> from coola.equality.handler import (
            ...     SameObjectHandler,
            ...     SameTypeHandler,
            ...     ObjectEqualHandler,
            ... )
            >>> config = EqualityConfig()
            >>> handler = SameObjectHandler()
            >>> handler.chain(SameTypeHandler()).chain(ObjectEqualHandler())
            >>> handler.handle([1, 2, 3], [1, 2, 3], config)
            True

            ```
        """
        self.set_next_handler(handler)
        return handler

    def chain_all(self, *handlers: BaseEqualityHandler) -> BaseEqualityHandler:
        r"""Chain multiple handlers in sequence.

        Args:
            *handlers: Variable number of handlers to chain.

        Returns:
            The last handler in the chain.

        Example:
            ```pycon
            >>> from coola.equality.handler import (
            ...     SameObjectHandler,
            ...     SameTypeHandler,
            ...     SameLengthHandler,
            ...     ObjectEqualHandler,
            ... )
            >>> handler = SameObjectHandler()
            >>> handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())

            ```
        """
        current = self
        for handler in handlers:
            current = current.chain(handler)
        return current

    def set_next_handler(self, handler: BaseEqualityHandler | None) -> None:
        r"""Set the next handler.

        Args:
            handler: The next handler. ``None`` means it is a terminal handler
                and there is no next handler.

        Example:
            ```pycon
            >>> from coola.equality.handler import SameObjectHandler, TrueHandler
            >>> handler = SameObjectHandler()
            >>> handler.set_next_handler(TrueHandler())

            ```
        """
        self._verify_next_handler(handler)
        self._next_handler = handler

    def _handle_next(self, actual: object, expected: object, config: EqualityConfig) -> bool:
        r"""Return the output from the next handler.

        Args:
            actual: The actual input.
            expected: The expected input.
            config: The equality configuration.

        Returns:
            The output from the next handler.

        Raises:
            RuntimeError: if the next handler is not defined.
        """
        if not self._next_handler:
            msg = "next handler is not defined"
            raise RuntimeError(msg)
        return self._next_handler.handle(actual, expected, config=config)

    def _verify_next_handler(self, handler: BaseEqualityHandler | None) -> None:
        if handler is None:
            return
        if handler is self:
            msg = "The current handler cannot be its next handler because it creates a cycle"
            raise RuntimeError(msg)
        if not isinstance(handler, BaseEqualityHandler):
            msg = (
                f"Incorrect type for `handler`. Expected "
                f"{BaseEqualityHandler.__qualname__} but received {type(handler)}"
            )
            raise TypeError(msg)

    def __repr__(self) -> str:
        if self.next_handler:
            return f"{self.__class__.__qualname__}(next_handler={self.next_handler})"
        return f"{self.__class__.__qualname__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

r"""Define the equality handler base classes."""

from __future__ import annotations

__all__ = ["BaseEqualityHandler"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from coola.utils.format import str_sequence

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

T = TypeVar("T", bound="BaseEqualityHandler")


class BaseEqualityHandler(ABC):
    r"""Define the base class to implement an equality handler.

    A child class needs to implement the ``handle`` method.
    The ``equal`` method has a default implementation that can be
    overridden for custom equality comparison (e.g., when handlers
    have additional state that needs to be compared).

    A terminal handler has its next handler set to ``None``.

    Args:
        next_handler: The next handler.

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

    def equal(self, other: object) -> bool:
        r"""Indicate if two objects are equal or not.

        This default implementation checks if both handlers are of the same
        type and have equal next handlers. Subclasses can override this
        method to add additional equality checks (e.g., comparing handler
        parameters).

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
        # Import here to avoid circular import at module level
        from coola.equality.handler.utils import handlers_are_equal

        if type(other) is not type(self):
            return False
        return handlers_are_equal(self.next_handler, other.next_handler)

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
        if handler is None:
            msg = "The next handler in the chain cannot be None."
            raise TypeError(msg)
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

    def get_chain_length(self) -> int:
        r"""Get the total number of handlers in the chain.

        Returns:
            The total number of handlers in the chain.

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
            >>> handler.get_chain_length()
            4

            ```
        """
        count = 1
        current = self.next_handler
        while current:
            count += 1
            current = current.next_handler
        return count

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

    def visualize_chain(self) -> str:
        r"""Visualize the current handler chain.

        Returns:
            A string containing the visualization of the current handler chain.

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
            >>> print(handler.visualize_chain())
            (0): SameObjectHandler()
            (1): SameTypeHandler()
            (2): SameLengthHandler()
            (3): ObjectEqualHandler()

            ```
        """
        handlers = []
        current = self
        while current:
            handlers.append(current)
            current = current.next_handler
        return str_sequence(handlers)

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
        r"""Verify the next handler is valid.

        Args:
            handler: The next handler.

        Raises:
            RuntimeError: if the next handler is self.
            TypeError: if the next handler is not a BaseEqualityHandler.
        """
        if not handler:
            return
        if handler is self:
            msg = "Cycle detected! the current handler cannot be its next handler"
            raise RuntimeError(msg)
        if not isinstance(handler, BaseEqualityHandler):
            msg = (
                f"Incorrect type for 'handler'. Expected "
                f"{BaseEqualityHandler} but received {type(handler)}"
            )
            raise TypeError(msg)

    def __repr__(self) -> str:
        if self.next_handler:
            return f"{self.__class__.__qualname__}(next_handler={self.next_handler!r})"
        return f"{self.__class__.__qualname__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}()"

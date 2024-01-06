r"""Define the equality handler base classes."""

from __future__ import annotations

__all__ = ["BaseEqualityHandler", "AbstractEqualityHandler"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class BaseEqualityHandler(ABC):
    r"""Define the base class to implement an equality handler.

    A child class needs to implement the following methods:

    - ``handle``
    - ``set_next_handler``

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameObjectHandler, FalseHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameObjectHandler()
    >>> handler.set_next_handler(FalseHandler())
    >>> handler.handle("abc", "abc", config)
    True
    >>> handler.handle("abc", "ABC", config)
    False

    ```
    """

    @abstractmethod
    def handle(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        r"""Return the equality result between the two input objects.

        Args:
            object1: Specifies the first object to compare.
            object2: Specifies the second object to compare.
            config: Specifies the equality configuration.

        Returns:
            ``True`` if the input objects are equal, and ``False``
                otherwise.

        Example usage:

        ```pycon
        >>> from coola.equality import EqualityConfig
        >>> from coola.equality.handlers import SameObjectHandler
        >>> from coola.testers import EqualityTester
        >>> config = EqualityConfig(tester=EqualityTester())
        >>> handler = SameObjectHandler()
        >>> handler.handle("abc", "abc", config)
        True

        ```
        """

    @abstractmethod
    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        r"""Set the next handler.

        Args:
            handler: The next handler.

        Example usage:

        ```pycon
        >>> from coola.equality.handlers import SameObjectHandler, TrueHandler
        >>> handler = SameObjectHandler()
        >>> handler.set_next_handler(TrueHandler())

        ```
        """


class AbstractEqualityHandler(BaseEqualityHandler):
    r"""Implement a base class with the default chaining behavior.

    A child class needs to implement the following method: ``handle``.
    """

    def __init__(self, next_handler: BaseEqualityHandler | None = None) -> None:
        self._next_handler = None
        if next_handler:
            self.set_next_handler(next_handler)

    @property
    def next_handler(self) -> BaseEqualityHandler | None:
        """The next handler."""
        return self._next_handler

    def _handle_next(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        r"""Return the output from the next handler.

        Returns:
            The output from the next handler.

        Raises:
            RuntimeError: if the next handler is not defined.
        """
        if not self._next_handler:
            msg = "The next handler is not defined"
            raise RuntimeError(msg)
        return self._next_handler.handle(object1=object1, object2=object2, config=config)

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        if not isinstance(handler, BaseEqualityHandler):
            msg = (
                f"Incorrect type for `handler`. Expected "
                f"{BaseEqualityHandler} but received {type(handler)}"
            )
            raise TypeError(msg)
        self._next_handler = handler
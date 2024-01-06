r"""Implement some handlers for native python objects."""

from __future__ import annotations

__all__ = [
    "FalseHandler",
    "SameObjectHandler",
    "SameTypeHandler",
    "TrueHandler",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class FalseHandler(BaseEqualityHandler):
    r"""Implement a handler that always return ``False``.

    This handler is designed to be used at the end of the chain of
    responsibility.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import FalseHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = FalseHandler()
    >>> handler.handle("abc", "abc", config)
    False
    >>> handler.handle("abc", "ABC", config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: Any,
        object2: Any,
        config: EqualityConfig,
    ) -> bool | None:
        return False

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class TrueHandler(BaseEqualityHandler):
    r"""Implement a handler that always return ``True``.

    This handler is designed to be used at the end of the chain of
    responsibility.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import TrueHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = TrueHandler()
    >>> handler.handle("abc", "abc", config)
    True
    >>> handler.handle("abc", "ABC", config)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: Any,
        object2: Any,
        config: EqualityConfig,
    ) -> bool | None:
        return True

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class SameObjectHandler(AbstractEqualityHandler):
    r"""Check if the two objects refer to the same object.

    This handler returns ``True`` if the two objects refer to the
    same object, otherwise it passes the inputs to the next handler.

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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: Any,
        object2: Any,
        config: EqualityConfig,
    ) -> bool | None:
        if object1 is object2:
            return True
        return self._handle_next(object1=object1, object2=object2, config=config)


class SameTypeHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same type.

    This handler returns ``False`` if the two objects have different
    types, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameTypeHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameTypeHandler()
    >>> handler.handle(1, "abc", config)
    False


    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: Any,
        object2: Any,
        config: EqualityConfig,
    ) -> bool | None:
        if type(object1) is not type(object2):
            if config.show_difference:
                logger.info(f"The objects have different types: {type(object1)} vs {type(object2)}")
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)

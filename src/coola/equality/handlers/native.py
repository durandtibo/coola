r"""Implement some handlers for native python objects."""

from __future__ import annotations

__all__ = [
    "FalseHandler",
    "ObjectEqualHandler",
    "SameKeysHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameTypeHandler",
    "TrueHandler",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler

if TYPE_CHECKING:
    from collections.abc import Mapping, Sized

    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class FalseHandler(BaseEqualityHandler):
    r"""Implement a handler that always return ``False``.

    This handler is designed to be used at the end of the chain of
    responsibility. This handler does not call the next handler.

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
    responsibility. This handler does not call the next handler.

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


class ObjectEqualHandler(BaseEqualityHandler):
    r"""Check if the two objects are equal using the default equality
    operator ``==``.

    This handler returns ``True`` if the two objects are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import ObjectEqualHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = ObjectEqualHandler()
    >>> handler.handle(1, 1, config)
    True
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
        object_equal = object1 == object2
        if config.show_difference and not object_equal:
            logger.info(f"objects are different:\nobject1={object1}\nobject2={object2}")
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class SameKeysHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same keys.

    This handler returns ``False`` if the two objects have different
    keys, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameKeysHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameKeysHandler()
    >>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 1}, config)
    False


    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: Mapping,
        object2: Mapping,
        config: EqualityConfig,
    ) -> bool | None:
        if set(object1.keys()) != set(object2.keys()):
            if config.show_difference:
                logger.info(
                    f"objects have different keys:\n"
                    f"object1 keys: {sorted(set(object1.keys()))}\n"
                    f"object2 keys: {sorted(set(object2.keys()))}"
                )
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)


class SameLengthHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same length.

    This handler returns ``False`` if the two objects have different
    lengths, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameLengthHandler
    >>> from coola.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameLengthHandler()
    >>> handler.handle([1, 2, 3], [1, 2, 3, 4], config)
    False


    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        object1: Sized,
        object2: Sized,
        config: EqualityConfig,
    ) -> bool | None:
        if len(object1) != len(object2):
            if config.show_difference:
                logger.info(f"objects have different lengths: {len(object1):,} vs {len(object2):,}")
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)


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
                logger.info(f"objects have different types: {type(object1)} vs {type(object2)}")
            return False
        return self._handle_next(object1=object1, object2=object2, config=config)

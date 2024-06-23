r"""Implement some handlers for native python objects."""

from __future__ import annotations

__all__ = [
    "FalseHandler",
    "ObjectEqualHandler",
    "SameAttributeHandler",
    "SameLengthHandler",
    "SameObjectHandler",
    "SameTypeHandler",
    "TrueHandler",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler
from coola.utils import repr_indent, repr_mapping

if TYPE_CHECKING:
    from collections.abc import Sized

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
    >>> from coola.equality.testers import EqualityTester
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
        actual: Any,  # noqa: ARG002
        expected: Any,  # noqa: ARG002
        config: EqualityConfig,  # noqa: ARG002
    ) -> bool:
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
    >>> from coola.equality.testers import EqualityTester
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
        actual: Any,  # noqa: ARG002
        expected: Any,  # noqa: ARG002
        config: EqualityConfig,  # noqa: ARG002
    ) -> bool:
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
    >>> from coola.equality.testers import EqualityTester
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
        actual: Any,
        expected: Any,
        config: EqualityConfig,
    ) -> bool:
        object_equal = actual == expected
        if config.show_difference and not object_equal:
            logger.info(f"objects are different:\nactual={actual}\nexpected={expected}")
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class SameAttributeHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same attribute.

    This handler returns ``False`` if the two objects have different
    attributes, otherwise it passes the inputs to the next handler.
    The objects must have the attribute.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameAttributeHandler, TrueHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameAttributeHandler(name="shape", next_handler=TrueHandler())
    >>> handler.handle(np.ones((2, 3)), np.ones((2, 3)), config)
    True
    >>> handler.handle(np.ones((2, 3)), np.ones((3, 2)), config)
    False

    ```
    """

    def __init__(self, name: str, next_handler: BaseEqualityHandler | None = None) -> None:
        super().__init__(next_handler=next_handler)
        self._name = name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"name": self._name, "next_handler": self._next_handler}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(name={self._name})"

    @property
    def name(self) -> str:
        return self._name

    def handle(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        value1 = getattr(actual, self._name)
        value2 = getattr(expected, self._name)
        if not config.tester.equal(value1, value2, config):
            if config.show_difference:
                logger.info(f"objects have different {self._name}: {value1} vs {value2}")
            return False
        return self._handle_next(actual, expected, config=config)


class SameLengthHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same length.

    This handler returns ``False`` if the two objects have different
    lengths, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon

    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameLengthHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameLengthHandler()
    >>> handler.handle([1, 2, 3], [1, 2, 3, 4], config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self,
        actual: Sized,
        expected: Sized,
        config: EqualityConfig,
    ) -> bool:
        if len(actual) != len(expected):
            if config.show_difference:
                logger.info(f"objects have different lengths: {len(actual):,} vs {len(expected):,}")
            return False
        return self._handle_next(actual, expected, config=config)


class SameObjectHandler(AbstractEqualityHandler):
    r"""Check if the two objects refer to the same object.

    This handler returns ``True`` if the two objects refer to the
    same object, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon

    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameObjectHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameObjectHandler()
    >>> handler.handle("abc", "abc", config)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self,
        actual: Any,
        expected: Any,
        config: EqualityConfig,
    ) -> bool | None:
        if actual is expected:
            return True
        return self._handle_next(actual, expected, config=config)


class SameTypeHandler(AbstractEqualityHandler):
    r"""Check if the two objects have the same type.

    This handler returns ``False`` if the two objects have different
    types, otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon

    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import SameTypeHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = SameTypeHandler()
    >>> handler.handle(1, "abc", config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self,
        actual: Any,
        expected: Any,
        config: EqualityConfig,
    ) -> bool:
        if type(actual) is not type(expected):
            if config.show_difference:
                logger.info(f"objects have different types: {type(actual)} vs {type(expected)}")
            return False
        return self._handle_next(actual, expected, config=config)

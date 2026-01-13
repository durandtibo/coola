r"""Implement some handler for native python objects."""

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
from typing import TYPE_CHECKING

from coola.equality.handler.base import AbstractEqualityHandler, BaseEqualityHandler
from coola.utils.format import repr_indent, repr_mapping

if TYPE_CHECKING:
    from collections.abc import Sized

    from coola.equality.config import EqualityConfig2

logger: logging.Logger = logging.getLogger(__name__)


class FalseHandler(BaseEqualityHandler):  # noqa: PLW1641
    r"""Implement a handler that always returns ``False``.

    This handler is designed to be used at the end of the chain of
    responsibility. This handler does not call the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import FalseHandler
        >>> config = EqualityConfig2()
        >>> handler = FalseHandler()
        >>> handler.handle("abc", "abc", config)
        False
        >>> handler.handle("abc", "ABC", config)
        False

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        actual: object,  # noqa: ARG002
        expected: object,  # noqa: ARG002
        config: EqualityConfig2,  # noqa: ARG002
    ) -> bool:
        return False

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class TrueHandler(BaseEqualityHandler):  # noqa: PLW1641
    r"""Implement a handler that always returns ``True``.

    This handler is designed to be used at the end of the chain of
    responsibility. This handler does not call the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import TrueHandler
        >>> config = EqualityConfig2()
        >>> handler = TrueHandler()
        >>> handler.handle("abc", "abc", config)
        True
        >>> handler.handle("abc", "ABC", config)
        True

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        actual: object,  # noqa: ARG002
        expected: object,  # noqa: ARG002
        config: EqualityConfig2,  # noqa: ARG002
    ) -> bool:
        return True

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class ObjectEqualHandler(BaseEqualityHandler):  # noqa: PLW1641
    r"""Check if the two objects are equal using the default equality
    operator ``==``.

    This handler returns ``True`` if the two objects are equal,
    otherwise ``False``. This handler is designed to be used at
    the end of the chain of responsibility. This handler does
    not call the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import ObjectEqualHandler
        >>> config = EqualityConfig2()
        >>> handler = ObjectEqualHandler()
        >>> handler.handle(1, 1, config)
        True
        >>> handler.handle(1, "abc", config)
        False

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        object_equal = actual == expected
        if config.show_difference and not object_equal:
            logger.info(f"objects are different:\nactual={actual}\nexpected={expected}")
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class SameAttributeHandler(AbstractEqualityHandler):  # noqa: PLW1641
    r"""Check if the two objects have the same attribute.

    This handler returns ``False`` if the two objects have different
    attributes, otherwise it passes the inputs to the next handler.
    The objects must have the attribute.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import SameAttributeHandler, TrueHandler
        >>> config = EqualityConfig2()
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
        if type(other) is not type(self):
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

    def handle(self, actual: object, expected: object, config: EqualityConfig2) -> bool:
        value1 = getattr(actual, self._name)
        value2 = getattr(expected, self._name)
        if not config.registry.objects_are_equal(value1, value2, config):
            if config.show_difference:
                logger.info(f"objects have different {self._name}: {value1} vs {value2}")
            return False
        return self._handle_next(actual, expected, config=config)


class SameLengthHandler(AbstractEqualityHandler):  # noqa: PLW1641
    r"""Check if the two objects have the same length.

    This handler returns ``False`` if the two objects have different
    lengths, otherwise it passes the inputs to the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import SameLengthHandler
        >>> config = EqualityConfig2()
        >>> handler = SameLengthHandler()
        >>> handler.handle([1, 2, 3], [1, 2, 3, 4], config)
        False

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def handle(
        self,
        actual: Sized,
        expected: Sized,
        config: EqualityConfig2,
    ) -> bool:
        if len(actual) != len(expected):
            if config.show_difference:
                logger.info(f"objects have different lengths: {len(actual):,} vs {len(expected):,}")
            return False
        return self._handle_next(actual, expected, config=config)


class SameObjectHandler(AbstractEqualityHandler):  # noqa: PLW1641
    r"""Check if the two objects refer to the same object.

    This handler returns ``True`` if the two objects refer to the
    same object, otherwise it passes the inputs to the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import SameObjectHandler
        >>> config = EqualityConfig2()
        >>> handler = SameObjectHandler()
        >>> handler.handle("abc", "abc", config)
        True

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def handle(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        if actual is expected:
            return True
        return self._handle_next(actual, expected, config=config)


class SameTypeHandler(AbstractEqualityHandler):  # noqa: PLW1641
    r"""Check if the two objects have the same type.

    This handler returns ``False`` if the two objects have different
    types, otherwise it passes the inputs to the next handler.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig2
        >>> from coola.equality.handler import SameTypeHandler
        >>> config = EqualityConfig2()
        >>> handler = SameTypeHandler()
        >>> handler.handle(1, "abc", config)
        False

        ```
    """

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def handle(
        self,
        actual: object,
        expected: object,
        config: EqualityConfig2,
    ) -> bool:
        if type(actual) is not type(expected):
            if config.show_difference:
                logger.info(f"objects have different types: {type(actual)} vs {type(expected)}")
            return False
        return self._handle_next(actual, expected, config=config)

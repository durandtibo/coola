r"""Implement handlers to check the objects are equal."""

from __future__ import annotations

__all__ = [
    "EqualHandler",
    "EqualNanHandler",
    "SupportsEqual",
    "SupportsEqualNan",
]

import logging
from typing import TYPE_CHECKING, Any, Protocol

from coola.equality.handlers.base import BaseEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class SupportsEqual(Protocol):
    r"""Implement a protocol to represent objects with a ``equal``
    method."""

    def equal(self, other: Any) -> bool:
        r"""Return ``True`` if the two objects are equal, otherwise
        ``False``.

        Args:
            other: The value to compare with.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``
        """


class SupportsEqualNan(Protocol):
    r"""Implement a protocol to represent objects with a ``equal``
    method with an option compare NaNs."""

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Return ``True`` if the two objects are equal, otherwise
        ``False``.

        Args:
            other: The value to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``
        """


class EqualHandler(BaseEqualityHandler):
    r"""Check if the two objects have the same data.

    This handler returns ``False`` if the two objects are different
    data, otherwise it returns ``True``. The first object must have
    a ``equal`` attribute which indicates if the two objects are
    equal or not.

    Example usage:

    ```pycon

    >>> import math
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import EqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> class MyFloat:
    ...     def __init__(self, value: float) -> None:
    ...         self._value = value
    ...     def equal(self, other: float) -> bool:
    ...         return self._value == other
    ...
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = EqualHandler()
    >>> handler.handle(MyFloat(42), 42, config)
    True
    >>> handler.handle(MyFloat(42), float("nan"), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(self, actual: SupportsEqual, expected: Any, config: EqualityConfig) -> bool:
        if not actual.equal(expected):
            if config.show_difference:
                logger.info(f"objects are not equal:\nactual:\n{actual}\nexpected:\n{expected}")
            return False
        return True

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


class EqualNanHandler(BaseEqualityHandler):
    r"""Check if the two objects have the same data.

    This handler returns ``False`` if the two objects are different
    data, otherwise it returns ``True``. The first object must have
    a ``equal`` attribute which indicates if the two objects are
    equal or not.

    Example usage:

    ```pycon

    >>> import math
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import EqualNanHandler
    >>> from coola.equality.testers import EqualityTester
    >>> class MyFloat:
    ...     def __init__(self, value: float) -> None:
    ...         self._value = value
    ...     def equal(self, other: float, equal_nan: bool = False) -> bool:
    ...         if equal_nan and math.isnan(self._value) and math.isnan(other):
    ...             return True
    ...         return self._value == other
    ...
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = EqualNanHandler()
    >>> handler.handle(MyFloat(42), 42, config)
    True
    >>> handler.handle(MyFloat(float("nan")), float("nan"), config)
    False
    >>> config.equal_nan = True
    >>> handler.handle(MyFloat(float("nan")), float("nan"), config)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(self, actual: SupportsEqualNan, expected: Any, config: EqualityConfig) -> bool:
        if not actual.equal(expected, equal_nan=config.equal_nan):
            if config.show_difference:
                logger.info(f"objects are not equal:\nactual:\n{actual}\nexpected:\n{expected}")
            return False
        return True

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.

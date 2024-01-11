r"""Implement handlers to check the objects are equal."""

from __future__ import annotations

__all__ = ["ScalarEqualHandler", "FloatEqualHandler"]

import logging
import math
from typing import TYPE_CHECKING

from coola.equality.handlers.base import BaseEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger = logging.getLogger(__name__)


class FloatEqualHandler(BaseEqualityHandler):
    r"""Check if the two float numbers are equal or not.

    This handler returns ``False`` if the two float numbers are
    different, otherwise it returns ``True``.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import FloatEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = FloatEqualHandler()
    >>> handler.handle(42.0, 42.0, config)
    True
    >>> handler.handle(float("nan"), float("nan"), config)
    False
    >>> config.equal_nan = True
    >>> handler.handle(float("nan"), float("nan"), config)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(self, object1: float, object2: float, config: EqualityConfig) -> bool:
        object_equal = self._compare_objects(object1, object2, config)
        if not object_equal and config.show_difference:
            logger.info(f"numbers are not equal:\nobject1:\n{object1}\nobject2:\n{object2}")
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.

    def _compare_objects(self, number1: float, number2: float, config: EqualityConfig) -> bool:
        r"""Indicate if the two numbers are equal or not.

        Args:
            number1: Specifies the first number to compare.
            number2: Specifies the second number to compare.
            config: Specifies the equality configuration.

        Returns:
            ``True``if the two numbers are equal, otherwise ``False``.
        """
        if config.equal_nan and math.isnan(number1) and math.isnan(number2):
            return True
        return number1 == number2


class ScalarEqualHandler(BaseEqualityHandler):
    r"""Check if the two float numbers are equal or not.

    This handler returns ``False`` if the two float numbers are
    different, otherwise it returns ``True``. It is possible to
    control the tolerance by using ``atol`` and ``rtol``.
    By default, the tolerances are set to 0.

    Example usage:

    ```pycon
    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import ScalarEqualHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = ScalarEqualHandler()
    >>> handler.handle(42.0, 42.0, config)
    True
    >>> handler.handle(float("nan"), float("nan"), config)
    False
    >>> config.equal_nan = True
    >>> handler.handle(float("nan"), float("nan"), config)
    True
    >>> config.atol = 1e-3
    >>> handler.handle(42.0, 42.0001, config)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(self, object1: float, object2: float, config: EqualityConfig) -> bool:
        object_equal = self._compare_objects(object1, object2, config)
        if not object_equal and config.show_difference:
            logger.info(f"numbers are not equal:\nobject1:\n{object1}\nobject2:\n{object2}")
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.

    def _compare_objects(self, number1: float, number2: float, config: EqualityConfig) -> bool:
        r"""Indicate if the two numbers are equal within a tolerance.

        Args:
            number1: Specifies the first number to compare.
            number2: Specifies the second number to compare.
            config: Specifies the equality configuration.

        Returns:
            ``True``if the two numbers are equal within a tolerance,
                otherwise ``False``.
        """
        if config.equal_nan and math.isnan(number1) and math.isnan(number2):
            return True
        if config.atol > 0.0 or config.rtol > 0.0:
            return math.isclose(number1, number2, abs_tol=config.atol, rel_tol=config.rtol)
        return number1 == number2

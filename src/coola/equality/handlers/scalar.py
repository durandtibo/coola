r"""Implement handlers to check the objects are equal."""

from __future__ import annotations

__all__ = ["NanEqualHandler", "ScalarEqualHandler"]

import logging
import math
from typing import TYPE_CHECKING

from coola.equality.handlers.base import AbstractEqualityHandler, BaseEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger = logging.getLogger(__name__)


class NanEqualHandler(AbstractEqualityHandler):
    r"""Check if the two NaNs are equal.

    This handler returns ``True`` if the two numbers are NaNs,
    otherwise it passes the inputs to the next handler.

    Example usage:

    ```pycon

    >>> from coola.equality import EqualityConfig
    >>> from coola.equality.handlers import NanEqualHandler, FalseHandler
    >>> from coola.equality.testers import EqualityTester
    >>> config = EqualityConfig(tester=EqualityTester())
    >>> handler = NanEqualHandler(next_handler=FalseHandler())
    >>> handler.handle(float("nan"), float("nan"), config)
    False
    >>> config.equal_nan = True
    >>> handler.handle(float("nan"), float("nan"), config)
    True

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def handle(
        self,
        actual: float,
        expected: float,
        config: EqualityConfig,
    ) -> bool:
        if config.equal_nan and math.isnan(actual) and math.isnan(expected):
            return True
        return self._handle_next(actual, expected, config=config)


class ScalarEqualHandler(BaseEqualityHandler):
    r"""Check if the two numbers are equal or not.

    This handler returns ``False`` if the two numbers are
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
    >>> config.atol = 1e-3
    >>> handler.handle(42.0, 42.0001, config)
    True
    >>> handler.handle(float("nan"), float("nan"), config)
    False

    ```
    """

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def handle(self, actual: float, expected: float, config: EqualityConfig) -> bool:
        object_equal = number_equal(actual, expected, config)
        if not object_equal and config.show_difference:
            logger.info(f"numbers are not equal:\nactual:\n{actual}\nexpected:\n{expected}")
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.


def number_equal(number1: float, number2: float, config: EqualityConfig) -> bool:
    r"""Indicate if the two numbers are equal within a tolerance.

    Args:
        number1: The first number to compare.
        number2: The second number to compare.
        config: The equality configuration.

    Returns:
        ``True``if the two numbers are equal within a tolerance,
            otherwise ``False``.
    """
    if config.atol > 0.0 or config.rtol > 0.0:
        return math.isclose(number1, number2, abs_tol=config.atol, rel_tol=config.rtol)
    return number1 == number2

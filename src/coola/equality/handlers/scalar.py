r"""Implement handlers to check the objects are equal."""

from __future__ import annotations

__all__ = ["FloatEqualHandler"]

import logging
import math
from typing import TYPE_CHECKING, Any

from coola.equality.handlers.base import BaseEqualityHandler

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


logger = logging.getLogger(__name__)


class FloatEqualHandler(BaseEqualityHandler):
    r"""Check if the two float numbers are the same.

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

    def handle(self, object1: float, object2: Any, config: EqualityConfig) -> bool:
        if config.equal_nan and math.isnan(object1) and math.isnan(object2):
            object_equal = True
        else:
            object_equal = object1 == object2
        if not object_equal and config.show_difference:
            logger.info(f"numbers are not equal:\nobject1:\n{object1}\nobject2:\n{object2}")
        return object_equal

    def set_next_handler(self, handler: BaseEqualityHandler) -> None:
        pass  # Do nothing because the next handler is never called.

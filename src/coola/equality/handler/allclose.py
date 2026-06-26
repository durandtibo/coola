r"""Implement handler to check if the objects are equal."""

from __future__ import annotations

__all__ = ["AllCloseNanHandler", "SupportsAllCloseNan"]

import logging
from typing import TYPE_CHECKING, Protocol

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger: logging.Logger = logging.getLogger(__name__)


class SupportsAllCloseNan(Protocol):
    r"""Implement a protocol to represent objects with a ``allclose``
    method with an option compare NaNs."""

    def allclose(
        self,
        other: object,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        r"""Return ``True`` if the two objects are equal, otherwise
        ``False``.

        Args:
            other: The value to compare with.
            rtol: The relative tolerance parameter. Must be non-negative.
            atol: The absolute tolerance parameter. Must be non-negative.
            equal_nan: If ``True``, then two ``NaN``s  will be considered
                as equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.
        """


class AllCloseNanHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects are close in value, with optional NaN
    equality.

    This handler returns ``False`` if the two objects differ beyond
    tolerance, otherwise it returns True. The first object must
    implement an ``allclose`` method.

    Example:
        ```pycon
        >>> import math
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler import AllCloseNanHandler
        >>> class MyFloat:
        ...     def __init__(self, value: float) -> None:
        ...         self._value = float(value)
        ...     def allclose(
        ...         self,
        ...         other: object,
        ...         rtol: float = 1e-5,
        ...         atol: float = 1e-8,
        ...         equal_nan: bool = False,
        ...     ) -> bool:
        ...         if type(other) is not type(self):
        ...             return False
        ...         if equal_nan and math.isnan(self._value) and math.isnan(other._value):
        ...             return True
        ...         return self._value == other._value
        ...
        >>> config = EqualityConfig()
        >>> handler = AllCloseNanHandler()
        >>> handler.handle(MyFloat(42), MyFloat(42), config)
        True
        >>> handler.handle(MyFloat(42), 42, config)
        False
        >>> handler.handle(MyFloat(float("nan")), MyFloat(float("nan")), config)
        False
        >>> config.equal_nan = True
        >>> handler.handle(MyFloat(float("nan")), MyFloat(float("nan")), config)
        True

        ```
    """

    def handle(self, actual: SupportsAllCloseNan, expected: object, config: EqualityConfig) -> bool:
        if not hasattr(actual, "allclose") or not callable(actual.allclose):
            return False
        if not actual.allclose(
            expected, rtol=config.rtol, atol=config.atol, equal_nan=config.equal_nan
        ):
            if config.show_difference:
                logger.info(format_value_difference(actual=actual, expected=expected))
            return False
        return True

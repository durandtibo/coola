r"""Implement a combined handler that dispatches to ``allclose`` or
``equal``."""

from __future__ import annotations

__all__ = ["SupportsTolerantEqual", "TolerantEqualHandler"]

import logging
from typing import TYPE_CHECKING, Protocol

from coola.equality.handler.base import BaseEqualityHandler
from coola.equality.handler.format import format_value_difference
from coola.equality.handler.mixin import HandlerEqualityMixin

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig

logger = logging.getLogger(__name__)


class SupportsTolerantEqual(Protocol):
    r"""Implement a protocol to represent objects that support both an
    ``allclose`` method with tolerance and NaN options, and an ``equal``
    method with a NaN option."""

    def allclose(
        self,
        other: object,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
    ) -> bool:
        r"""Return ``True`` if the two objects are equal within
        tolerance, otherwise ``False``.

        Args:
            other: The value to compare with.
            rtol: The relative tolerance parameter. Must be non-negative.
            atol: The absolute tolerance parameter. Must be non-negative.
            equal_nan: If ``True``, then two ``NaN``s will be considered
                as equal.

        Returns:
            ``True`` if the two objects are equal within tolerance,
            otherwise ``False``.
        """

    def equal(self, other: object, equal_nan: bool = False) -> bool:
        r"""Return ``True`` if the two objects are equal, otherwise
        ``False``.

        Args:
            other: The value to compare with.
            equal_nan: If ``True``, then two ``NaN``s will be considered
                as equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.
        """


class TolerantEqualHandler(HandlerEqualityMixin, BaseEqualityHandler):
    r"""Check if the two objects are equal, dispatching to ``allclose``
    when tolerances are non-zero, otherwise to ``equal``.

    This handler returns ``False`` if the two objects differ, otherwise
    it returns ``True``. The first object must implement both an
    ``allclose`` method and an ``equal`` method.

    When ``config.atol`` or ``config.rtol`` is non-zero, ``allclose``
    is called with the configured tolerances and ``equal_nan``. When
    both are zero, ``equal`` is called with ``equal_nan`` only.

    Example:
        ```pycon
        >>> import math
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.handler.tolerant_equal import TolerantEqualHandler
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
        ...         return math.isclose(self._value, other._value, rel_tol=rtol, abs_tol=atol)
        ...     def equal(self, other: object, equal_nan: bool = False) -> bool:
        ...         if type(other) is not type(self):
        ...             return False
        ...         if equal_nan and math.isnan(self._value) and math.isnan(other._value):
        ...             return True
        ...         return self._value == other._value
        ...
        >>> config = EqualityConfig()
        >>> handler = TolerantEqualHandler()
        >>> handler.handle(MyFloat(42), MyFloat(42), config)
        True
        >>> handler.handle(MyFloat(42), 42, config)
        False
        >>> handler.handle(MyFloat(float("nan")), MyFloat(float("nan")), config)
        False
        >>> config.equal_nan = True
        >>> handler.handle(MyFloat(float("nan")), MyFloat(float("nan")), config)
        True
        >>> config.atol = 0.5
        >>> handler.handle(MyFloat(1.0), MyFloat(1.4), config)
        True

        ```
    """

    def handle(
        self, actual: SupportsTolerantEqual, expected: object, config: EqualityConfig
    ) -> bool:
        if (
            not hasattr(actual, "allclose")
            or not callable(actual.allclose)
            or not hasattr(actual, "equal")
            or not callable(actual.equal)
        ):
            return False
        if config.atol != 0 or config.rtol != 0:
            result = actual.allclose(
                expected, rtol=config.rtol, atol=config.atol, equal_nan=config.equal_nan
            )
        else:
            result = actual.equal(expected, equal_nan=config.equal_nan)
        if not result:
            if config.show_difference:
                logger.info(format_value_difference(actual=actual, expected=expected))
            return False
        return True

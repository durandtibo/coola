r"""Implement scalar equality testers.

This module provides equality testers for scalar numeric types (int, float)
with support for NaN equality and tolerance-based comparisons.
"""

from __future__ import annotations

__all__ = ["ScalarEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    NanEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    ScalarEqualHandler,
)
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class ScalarEqualityTester(BaseEqualityTester[float]):
    r"""Implement a scalar equality tester.

    This tester handles numeric scalar types (int, float) with support for
    NaN equality and tolerance-based comparisons. The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify same type
    3. NanEqualHandler: Handle NaN comparisons based on config.equal_nan
    4. ScalarEqualHandler: Compare values with tolerance (config.atol, config.rtol)

    This tester is registered for both int and float types in the default registry.

    Example:
        Basic scalar comparison:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import ScalarEqualityTester
        >>> config = EqualityConfig()
        >>> tester = ScalarEqualityTester()
        >>> tester.objects_are_equal(42.0, 42.0, config)
        True
        >>> tester.objects_are_equal(42.0, 1.0, config)
        False

        ```

        NaN comparison with equal_nan enabled:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import ScalarEqualityTester
        >>> config = EqualityConfig(equal_nan=True)
        >>> tester = ScalarEqualityTester()
        >>> tester.objects_are_equal(float("nan"), float("nan"), config)
        True

        ```

        Tolerance-based comparison:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import ScalarEqualityTester
        >>> config = EqualityConfig(atol=1e-6)
        >>> tester = ScalarEqualityTester()
        >>> tester.objects_are_equal(1.0, 1.0000001, config)
        True

        ```
    """

    def __init__(self) -> None:
        """Initialize the scalar equality tester with its handler chain.

        The handler chain performs checks in this order:
        1. SameObjectHandler: Quick check for object identity
        2. SameTypeHandler: Verify both objects have the same type
        3. NanEqualHandler: Handle NaN equality based on config.equal_nan
        4. ScalarEqualHandler: Compare numeric values with tolerance
        """
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(NanEqualHandler()).chain(ScalarEqualHandler())

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: object) -> bool:
        return type(other) is type(self)

    def objects_are_equal(
        self,
        actual: float,
        expected: object,
        config: EqualityConfig,
    ) -> bool:
        return self._handler.handle(actual, expected, config=config)

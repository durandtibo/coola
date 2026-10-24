r"""Implement an equality tester for the values that are compared with
their difference.

This module provides an equality tester for ``decimal.Decimal``,
``fractions.Fraction`` and ``datetime.timedelta``.
"""

from __future__ import annotations

__all__ = ["DifferenceCloseEqualityTester"]

from typing import TYPE_CHECKING, Any

from coola.equality.handler import (
    DifferenceCloseHandler,
    SameObjectHandler,
    SameTypeHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler


class DifferenceCloseEqualityTester(BaseChainEqualityTester[Any]):
    r"""Implement an equality tester for ``decimal.Decimal``,
    ``fractions.Fraction`` and ``datetime.timedelta``.

    These types have no ``allclose``, so they were compared with ``==``
    and ignored the tolerances. This tester compares their difference
    with the tolerance (see ``difference_close``). The handler chain:
    1. SameObjectHandler: Check for object identity
    2. SameTypeHandler: Verify same type
    3. DifferenceCloseHandler: Compare the difference with the tolerance

    This tester is registered for ``Decimal``, ``Fraction`` and
    ``timedelta`` in the default registry.

    Example:
        ```pycon
        >>> from decimal import Decimal
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import DifferenceCloseEqualityTester
        >>> tester = DifferenceCloseEqualityTester()
        >>> tester.objects_are_equal(Decimal("1.0"), Decimal("1.00001"), EqualityConfig())
        False
        >>> tester.objects_are_equal(Decimal("1.0"), Decimal("1.00001"), EqualityConfig(atol=1e-3))
        True

        ```
    """

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(SameObjectHandler(), SameTypeHandler(), DifferenceCloseHandler())

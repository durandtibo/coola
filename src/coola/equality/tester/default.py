r"""Implement the default equality tester.

This module provides a fallback equality tester for types that don't
have a specialized tester registered. It uses Python's built-in equality
operator.
"""

from __future__ import annotations

__all__ = ["DefaultEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality.handler import (
    ObjectEqualHandler,
    SameObjectHandler,
    SameTypeHandler,
    create_chain,
)
from coola.equality.tester.chain import BaseChainEqualityTester

if TYPE_CHECKING:
    from coola.equality.handler import BaseEqualityHandler


class DefaultEqualityTester(BaseChainEqualityTester[object]):
    r"""Implement a default equality tester.

    This tester serves as the fallback for types without specialized equality
    testers. It uses Python's built-in ``==`` operator to test equality between
    objects. The tester uses a handler chain to perform checks in order:
    1. SameObjectHandler: Check if objects are the same instance (identity)
    2. SameTypeHandler: Check if objects have the same type
    3. ObjectEqualHandler: Use ``==`` operator for equality comparison

    This tester is registered for the `object` type in the default registry,
    making it the catch-all for unregistered types via Python's MRO.

    Example:
        Basic usage with primitives:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import DefaultEqualityTester
        >>> config = EqualityConfig()
        >>> tester = DefaultEqualityTester()
        >>> tester.objects_are_equal(42, 42, config=config)
        True
        >>> tester.objects_are_equal("meow", "meov", config)
        False

        ```

        Different types are not equal:

        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.tester import DefaultEqualityTester
        >>> config = EqualityConfig()
        >>> tester = DefaultEqualityTester()
        >>> tester.objects_are_equal(42, "42", config)
        False

        ```
    """

    def _create_handler(self) -> BaseEqualityHandler:
        return create_chain(SameObjectHandler(), SameTypeHandler(), ObjectEqualHandler())

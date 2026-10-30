r"""Implement the default equality tester.

This module provides the fallback tester for types without a specialized
tester. It compares the objects with the equality operator ``==``.
"""

from __future__ import annotations

__all__ = ["DefaultEqualityTester"]

from typing import TYPE_CHECKING

from coola.equality._helpers.format import format_value_difference
from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from coola.equality.engine import Engine


class DefaultEqualityTester(BaseEqualityTester[object]):
    r"""Implement a default equality tester.

    This tester serves as the fallback for types without specialized
    equality testers. It uses Python's built-in ``==`` operator to test
    equality between two objects of the same type. The engine checks
    the identity and the type before the tester is called.

    This tester is registered for the ``object`` type in the default
    registry, making it the catch-all for unregistered types via
    Python's MRO.

    Raises:
        TypeError: if ``==`` raises a ``TypeError`` or a ``ValueError``,
            or returns a value that cannot be converted to a ``bool``
            (for example an array). ``False`` would claim that the
            objects are different when they were not compared at all.

    Example:
        ```pycon
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import DefaultEqualityTester, get_default_registry
        >>> engine = Engine(EqualityConfig())
        >>> tester = DefaultEqualityTester()
        >>> tester.objects_are_equal("meow", "meow", engine)
        True
        >>> tester.objects_are_equal("meow", "meov", engine)
        False

        ```
    """

    def objects_are_equal(self, actual: object, expected: object, ctx: Engine) -> bool:
        r"""Test two objects with ``==``.

        Args:
            actual: The actual object.
            expected: The expected object, with the same type.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the objects are equal, otherwise ``False`` (the
                reason is recorded in the engine).

        Raises:
            TypeError: If ``==`` raises or returns a value that cannot
                be converted to a ``bool``.
        """
        try:
            equal = bool(actual == expected)
        except (TypeError, ValueError) as exc:
            # e.g. a dataclass holding an array: ``==`` returns an array, or ``==`` itself
            # raises (a pydantic model holding an array). A ``TypeError`` is raised
            # instead of returning ``False``, because ``False`` would claim that the
            # objects are different when they were not compared at all.
            msg = (
                f"`==` raised or returned a value that cannot be converted to a bool "
                f"({exc}) when comparing {type(actual)} objects. Register an equality "
                "tester for this type."
            )
            raise TypeError(msg) from exc
        return equal or ctx.fail(lambda: format_value_difference(actual=actual, expected=expected))

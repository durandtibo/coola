r"""Define some testing utilities.

This module provides convenient assertion functions for testing code
that works with complex nested data structures.

Examples:
    >>> from coola.testing import assert_equal
    >>> assert_equal([1, 2, 3], [1, 2, 3])  # Passes silently
    >>> assert_equal([1, 2], [1, 3])  # Raises AssertionError

    >>> from coola.testing import assert_allclose
    >>> assert_allclose([1.0, 2.0], [1.0 + 1e-9, 2.0])  # Passes
"""

from __future__ import annotations

__all__ = ["assert_allclose", "assert_equal"]

from typing import TYPE_CHECKING

from coola.equality import objects_are_allclose, objects_are_equal

if TYPE_CHECKING:
    from coola.equality.tester.registry import EqualityTesterRegistry


def assert_equal(
    actual: object,
    expected: object,
    *,
    equal_nan: bool = False,
    show_difference: bool = True,
    max_depth: int = 1000,
    registry: EqualityTesterRegistry | None = None,
) -> None:
    r"""Assert that two objects are exactly equal.

    This function is designed for use in test code. It raises an
    AssertionError if the objects are not equal, with a descriptive
    message showing the differences.

    Args:
        actual: The actual value to test.
        expected: The expected value.
        equal_nan: If ``True``, then two ``NaN``s will be considered
            as equal. Defaults to ``False``.
        show_difference: If ``True``, shows the difference between
            the two objects if they are different. Defaults to ``True``
            for better debugging experience.
        max_depth: Maximum recursion depth for nested comparisons.
            Must be positive. Defaults to 1000.
        registry: The registry with the equality tester to use.
            If ``None``, uses the default registry.

    Raises:
        AssertionError: if the two objects are not equal.
        ValueError: if ``max_depth`` is not positive.
        RecursionError: if recursion depth exceeds ``max_depth``.

    Example:
        ```pycon
        >>> from coola.testing import assert_equal
        >>> assert_equal([1, 2, 3], [1, 2, 3])
        >>> assert_equal({"a": [1, 2], "b": 3}, {"a": [1, 2], "b": 3})

        ```
    """
    if max_depth <= 0:
        msg = f"max_depth must be positive, but got {max_depth}"
        raise ValueError(msg)
    if not objects_are_equal(
        actual,
        expected,
        equal_nan=equal_nan,
        show_difference=show_difference,
        max_depth=max_depth,
        registry=registry,
    ):
        msg = f"Objects are not equal:\n  actual  : {actual!r}\n  expected: {expected!r}"
        raise AssertionError(msg)


def assert_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    show_difference: bool = True,
    max_depth: int = 1000,
    registry: EqualityTesterRegistry | None = None,
) -> None:
    r"""Assert that two objects are equal within a tolerance.

    This function is designed for use in test code. It raises an
    AssertionError if the objects are not approximately equal, with
    a descriptive message showing the differences.

    Args:
        actual: The actual value to test.
        expected: The expected value.
        rtol: The relative tolerance parameter. Must be non-negative.
            Defaults to 1e-5.
        atol: The absolute tolerance parameter. Must be non-negative.
            Defaults to 1e-8.
        equal_nan: If ``True``, then two ``NaN``s will be considered
            as equal. Defaults to ``False``.
        show_difference: If ``True``, shows the difference between
            the two objects if they are different. Defaults to ``True``
            for better debugging experience.
        max_depth: Maximum recursion depth for nested comparisons.
            Must be positive. Defaults to 1000.
        registry: The registry with the equality tester to use.
            If ``None``, uses the default registry.

    Raises:
        AssertionError: if the two objects are not approximately equal.
        ValueError: if ``rtol`` or ``atol`` is negative, or if
            ``max_depth`` is not positive.
        RecursionError: if recursion depth exceeds ``max_depth``.

    Example:
        ```pycon
        >>> from coola.testing import assert_allclose
        >>> assert_allclose([1.0, 2.0], [1.0 + 1e-9, 2.0])
        >>> assert_allclose({"a": 1.0}, {"a": 1.0 + 1e-7}, atol=1e-6)

        ```
    """
    if rtol < 0:
        msg = f"rtol must be non-negative, but got {rtol}"
        raise ValueError(msg)
    if atol < 0:
        msg = f"atol must be non-negative, but got {atol}"
        raise ValueError(msg)
    if max_depth <= 0:
        msg = f"max_depth must be positive, but got {max_depth}"
        raise ValueError(msg)
    if not objects_are_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        show_difference=show_difference,
        max_depth=max_depth,
        registry=registry,
    ):
        msg = (
            f"Objects are not approximately equal (rtol={rtol}, atol={atol}):\n"
            f"  actual  : {actual!r}\n"
            f"  expected: {expected!r}"
        )
        raise AssertionError(msg)

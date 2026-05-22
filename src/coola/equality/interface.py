r"""Define the public interface to compare nested data for equality."""

from __future__ import annotations

__all__ = ["objects_are_allclose", "objects_are_equal"]

from typing import TYPE_CHECKING

from coola.equality.config import EqualityConfig
from coola.equality.tester.interface import get_default_registry

if TYPE_CHECKING:
    from coola.equality.tester.registry import EqualityTesterRegistry


def objects_are_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    show_difference: bool = False,
    max_depth: int = 1000,
    registry: EqualityTesterRegistry | None = None,
) -> bool:
    r"""Indicate if two objects are equal within a tolerance.

    Args:
        actual: The actual object.
        expected: The expected object.
        rtol: The relative tolerance parameter. Must be non-negative.
        atol: The absolute tolerance parameter. Must be non-negative.
        equal_nan: If ``True``, treat two ``NaN`` values as equal.
        show_difference: If ``True``, log a readable description when the
            comparison fails.
        max_depth: Maximum recursion depth for nested comparisons.
            Must be positive. Defaults to 1000.
        registry: Registry used to resolve type-specific equality testers.
            If ``None``, the default registry is used.

    Returns:
        ``True`` if the two objects are equal within tolerances, otherwise
        ``False``.

    Raises:
        ValueError: If ``rtol`` or ``atol`` is negative.
        RecursionError: If recursion depth exceeds ``max_depth``.

    Example:
        ```pycon
        >>> from coola.equality import objects_are_allclose
        >>> objects_are_allclose([1.0, 2.0], [1.0 + 1e-8, 2.0])
        True
        >>> objects_are_allclose([1.0, 2.0], [1.0, 2.1], atol=1e-3, rtol=0.0)
        False
        >>> objects_are_allclose([float("nan")], [float("nan")], equal_nan=False)
        False
        >>> objects_are_allclose([float("nan")], [float("nan")], equal_nan=True)
        True

        ```
    """
    if rtol < 0:
        msg = f"rtol must be non-negative, but got {rtol}"
        raise ValueError(msg)
    if atol < 0:
        msg = f"atol must be non-negative, but got {atol}"
        raise ValueError(msg)
    if registry is None:
        registry = get_default_registry()
    config = EqualityConfig(
        registry=registry,
        show_difference=show_difference,
        equal_nan=equal_nan,
        atol=atol,
        rtol=rtol,
        max_depth=max_depth,
    )
    return registry.objects_are_equal(actual, expected, config)


def objects_are_equal(
    actual: object,
    expected: object,
    *,
    equal_nan: bool = False,
    show_difference: bool = False,
    max_depth: int = 1000,
    registry: EqualityTesterRegistry | None = None,
) -> bool:
    r"""Indicate if two objects are equal or not.

    Args:
        actual: The actual object.
        expected: The expected object.
        equal_nan: If ``True``, treat two ``NaN`` values as equal.
        show_difference: If ``True``, log a readable description when the
            comparison fails.
        max_depth: Maximum recursion depth for nested comparisons.
            Must be positive. Defaults to 1000.
        registry: Registry used to resolve type-specific equality testers.
            If ``None``, the default registry is used.

    Returns:
        ``True`` if the two objects are exactly equal, otherwise ``False``.

    Raises:
        RecursionError: If recursion depth exceeds ``max_depth``.

    Example:
        ```pycon
        >>> from coola.equality import objects_are_equal
        >>> objects_are_equal([1, {"a": 2}], [1, {"a": 2}])
        True
        >>> objects_are_equal([1, {"a": 2}], [1, {"a": 3}])
        False

        ```
    """
    if registry is None:
        registry = get_default_registry()
    config = EqualityConfig(
        registry=registry,
        show_difference=show_difference,
        equal_nan=equal_nan,
        max_depth=max_depth,
    )
    return registry.objects_are_equal(actual, expected, config)

r"""Define the public interface to recursively apply a function to all
items in nested data.
"""

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
        actual: The actual input.
        expected: The expected input.
        rtol: The relative tolerance parameter. Must be non-negative.
        atol: The absolute tolerance parameter. Must be non-negative.
        equal_nan: If ``True``, then two ``NaN``s  will be considered
            as equal.
        show_difference: If ``True``, it shows a difference between
            the two objects if they are different. This parameter is
            useful to find the difference between two objects.
        max_depth: Maximum recursion depth for nested comparisons.
            Must be positive. Defaults to 1000.
        registry: The registry with the equality tester to use.

    Returns:
        ``True`` if the two objects are (element-wise) equal within a
            tolerance, otherwise ``False``

    Raises:
        ValueError: if ``rtol`` or ``atol`` is negative.
        RecursionError: if recursion depth exceeds ``max_depth``.

    Example:
        ```pycon

        >>> import torch
        >>> from coola.equality import objects_are_allclose
        >>> objects_are_allclose(
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ... )
        True
        >>> objects_are_allclose(
        ...     [torch.ones(2, 3), torch.ones(2)],
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ... )
        False
        >>> objects_are_allclose(
        ...     [torch.ones(2, 3) + 1e-7, torch.ones(2)],
        ...     [torch.ones(2, 3), torch.ones(2) - 1e-7],
        ...     rtol=0,
        ...     atol=1e-8,
        ... )
        False

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
        actual: The actual input.
        expected: The expected input.
        equal_nan: If ``True``, then two ``NaN``s  will be considered
            as equal.
        show_difference: If ``True``, it shows a difference between
            the two objects if they are different. This parameter is
            useful to find the difference between two objects.
        max_depth: Maximum recursion depth for nested comparisons.
            Must be positive. Defaults to 1000.
        registry: The registry with the equality tester to use.

    Returns:
        ``True`` if the two nested data are equal, otherwise
            ``False``.

    Raises:
        RecursionError: if recursion depth exceeds ``max_depth``.

    Example:
        ```pycon

        >>> import torch
        >>> from coola.equality import objects_are_equal
        >>> objects_are_equal(
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ... )
        True
        >>> objects_are_equal([torch.ones(2, 3), torch.ones(2)], [torch.ones(2, 3), torch.zeros(2)])
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

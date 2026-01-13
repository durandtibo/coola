r"""Define the public interface to recursively apply a function to all
items in nested data."""

from __future__ import annotations

__all__ = ["objects_are_allclose", "objects_are_equal"]

from typing import TYPE_CHECKING

from coola.equality.config import EqualityConfig2

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
    registry: EqualityTesterRegistry | None = None,
) -> bool:
    r"""Indicate if two objects are equal within a tolerance.

    Args:
        actual: The actual input.
        expected: The expected input.
        rtol: The relative tolerance parameter.
        atol: The absolute tolerance parameter.
        equal_nan: If ``True``, then two ``NaN``s  will be considered
            as equal.
        show_difference: If ``True``, it shows a difference between
            the two objects if they are different. This parameter is
            useful to find the difference between two objects.
        registry: The registry with the equality tester to use.

    Returns:
        ``True`` if the two objects are (element-wise) equal within a
            tolerance, otherwise ``False``

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
    if registry is None:
        from coola.equality.tester.interface import get_default_registry

        registry = get_default_registry()
    config = EqualityConfig2(
        registry=registry,
        show_difference=show_difference,
        equal_nan=equal_nan,
        atol=atol,
        rtol=rtol,
    )
    return registry.objects_are_equal(actual, expected, config)


def objects_are_equal(
    actual: object,
    expected: object,
    *,
    equal_nan: bool = False,
    show_difference: bool = False,
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
        registry: The registry with the equality tester to use.

    Returns:
        ``True`` if the two nested data are equal, otherwise
            ``False``.

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
        from coola.equality.tester.interface import get_default_registry

        registry = get_default_registry()
    config = EqualityConfig2(
        registry=registry, show_difference=show_difference, equal_nan=equal_nan
    )
    return registry.objects_are_equal(actual, expected, config)

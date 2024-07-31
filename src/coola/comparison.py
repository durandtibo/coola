r"""Implement the main comparison public features."""

from __future__ import annotations

__all__ = ["objects_are_allclose", "objects_are_equal"]

from typing import TYPE_CHECKING, Any

from coola.equality.config import EqualityConfig
from coola.equality.testers import EqualityTester

if TYPE_CHECKING:
    from coola.equality.testers import BaseEqualityTester

_tester = EqualityTester()


def objects_are_allclose(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    show_difference: bool = False,
    tester: BaseEqualityTester | None = None,
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
        tester: The equality tester. If ``None``,
            ``EqualityTester`` is used.

    Returns:
        ``True`` if the two objects are (element-wise) equal within a
            tolerance, otherwise ``False``

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola import objects_are_allclose
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
    tester = tester or _tester
    config = EqualityConfig(
        tester=tester, show_difference=show_difference, equal_nan=equal_nan, atol=atol, rtol=rtol
    )
    return tester.equal(actual, expected, config)


def objects_are_equal(
    actual: Any,
    expected: Any,
    *,
    equal_nan: bool = False,
    show_difference: bool = False,
    tester: BaseEqualityTester | None = None,
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
        tester: The equality tester. If ``None``,
            ``EqualityTester`` is used.

    Returns:
        ``True`` if the two nested data are equal, otherwise
            ``False``.

    Example usage:

    ```pycon

    >>> import torch
    >>> from coola import objects_are_equal
    >>> objects_are_equal(
    ...     [torch.ones(2, 3), torch.zeros(2)],
    ...     [torch.ones(2, 3), torch.zeros(2)],
    ... )
    True
    >>> objects_are_equal([torch.ones(2, 3), torch.ones(2)], [torch.ones(2, 3), torch.zeros(2)])
    False

    ```
    """
    tester = tester or _tester
    config = EqualityConfig(tester=tester, show_difference=show_difference, equal_nan=equal_nan)
    return tester.equal(actual, expected, config)

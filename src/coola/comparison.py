from __future__ import annotations

__all__ = ["objects_are_equal"]

from typing import Any

from coola.testers import (
    AllCloseTester,
    BaseAllCloseTester,
    BaseEqualityTester,
    EqualityTester,
)


def objects_are_allclose(
    object1: Any,
    object2: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    show_difference: bool = False,
    tester: BaseAllCloseTester | None = None,
) -> bool:
    r"""Indicates if two objects are equal within a tolerance.

    Args:
        object1: Specifies the first object to compare.
        object2: Specifies the second object to compare.
        rtol (float, optional): Specifies the relative tolerance
            parameter. Default: ``1e-5``
        atol (float, optional): Specifies the absolute tolerance
            parameter. Default: ``1e-8``
        equal_nan (bool, optional): If ``True``, then two ``NaN``s
            will be considered equal. Default: ``False``
        show_difference (bool, optional): If ``True``, it shows a
            difference between the two objects if they are different.
            This parameter is useful to find the difference between
            two objects. Default: ``False``
        tester (``BaseAllCloseTester`` or ``None``, optional):
            Specifies an equality tester. If ``None``,
            ``AllCloseTester`` is used. Default: ``None``.

    Returns:
        bool: ``True`` if the two objects are (element-wise) equal
            within a tolerance, otherwise ``False``

    Example usage:

    .. code-block:: pycon

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
    """
    tester = tester or AllCloseTester()
    return tester.allclose(object1, object2, rtol, atol, equal_nan, show_difference)


def objects_are_equal(
    object1: Any,
    object2: Any,
    show_difference: bool = False,
    tester: BaseEqualityTester | None = None,
) -> bool:
    r"""Indicates if two objects are equal or not.

    Args:
        object1: Specifies the first object to compare.
        object2: Specifies the second object to compare.
        show_difference (bool, optional): If ``True``, it shows a
            difference between the two objects if they are
            different. This parameter is useful to find the
            difference between two objects. Default: ``False``
        tester (``BaseEqualityTester`` or ``None``, optional):
            Specifies an equality tester. If ``None``,
            ``EqualityTester`` is used. Default: ``None``.

    Returns:
        bool: ``True`` if the two nested data are equal, otherwise
            ``False``.

    Example usage:

    .. code-block:: pycon

        >>> import torch
        >>> from coola import objects_are_equal
        >>> objects_are_equal(
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ... )
        True
        >>> objects_are_equal([torch.ones(2, 3), torch.ones(2)], [torch.ones(2, 3), torch.zeros(2)])
        False
    """
    tester = tester or EqualityTester()
    return tester.equal(object1, object2, show_difference)

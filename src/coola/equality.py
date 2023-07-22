from __future__ import annotations

__all__ = ["objects_are_equal"]

import logging
from typing import Any, TypeVar

from coola.testers import BaseEqualityTester, EqualityTester

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

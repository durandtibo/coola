r"""Define the tester base class."""

from __future__ import annotations

__all__ = ["BaseAllCloseTester", "BaseEqualityTester"]

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseAllCloseTester(ABC):
    r"""Define the base class to implement an allclose tester."""

    @abstractmethod
    def allclose(
        self,
        object1: Any,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        r"""Indicate if two objects are equal within a tolerance.

        Args:
            object1: Specifies the first object to compare.
            object2: Specifies the second object to compare.
            rtol : Specifies the relative tolerance parameter.
            atol: Specifies the absolute tolerance parameter.
            equal_nan: If ``True``, then two ``NaN``s will be
                considered equal.
            show_difference: If ``True``, it shows a difference between
                the two objects if they are different. This parameter
                is useful to find the difference between two objects.

        Returns:
            ``True`` if the two objects are equal within a tolerance,
                otherwise ``False``

        Example usage:

        ```pycon
        >>> import torch
        >>> from coola import AllCloseTester, BaseAllCloseTester
        >>> tester: BaseAllCloseTester = AllCloseTester()
        >>> tester.allclose(
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ... )
        True
        >>> tester.allclose(
        ...     [torch.ones(2, 3), torch.ones(2)],
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ... )
        False
        >>> tester.allclose(
        ...     [torch.ones(2, 3) + 1e-7, torch.ones(2)],
        ...     [torch.ones(2, 3), torch.ones(2) - 1e-7],
        ...     rtol=0,
        ...     atol=1e-8,
        ... )
        False

        ```
        """


class BaseEqualityTester(ABC):
    r"""Defines the base class to implement an equality tester."""

    @abstractmethod
    def equal(self, object1: Any, object2: Any, show_difference: bool = False) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            object1: Specifies the first object to compare.
            object2: Specifies the second object to compare.
            show_difference: If ``True``, it shows a difference between
                the two objects if they are different. This parameter
                is useful to find the difference between two objects.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example usage:

        ```pycon
        >>> import torch
        >>> from coola import BaseEqualityTester, EqualityTester
        >>> tester: BaseEqualityTester = EqualityTester()
        >>> tester.equal(
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ... )
        True
        >>> tester.equal([torch.ones(2, 3), torch.ones(2)], [torch.ones(2, 3), torch.zeros(2)])
        False

        ```
        """

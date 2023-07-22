from __future__ import annotations

__all__ = ["BaseEqualityTester"]

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseEqualityTester(ABC):
    r"""Defines the base class to implement an equality tester."""

    @abstractmethod
    def equal(self, object1: Any, object2: Any, show_difference: bool = False) -> bool:
        r"""Indicates if two objects are equal or not.

        Args:
            object1: Specifies the first object to compare.
            object2: Specifies the second object to compare.
            show_difference (bool, optional): If ``True``, it shows a
                difference between the two objects if they are
                different. This parameter is useful to find the
                difference between two objects. Default: ``False``

        Returns:
            bool: ``True`` if the two objects are equal, otherwise
                ``False``.

        Example usage:

        .. code-block:: pycon

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
        """

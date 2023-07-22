from __future__ import annotations

__all__ = ["BaseEqualityOperator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from coola.testers import BaseEqualityTester

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseEqualityOperator(ABC, Generic[T]):
    r"""Define the base class to implement an equality operator."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @abstractmethod
    def clone(self) -> BaseEqualityOperator:
        r"""Returns a copy of the equality operator.

        Returns:
            ``BaseEqualityOperator``: A copy of the equality operator.
        """

    @abstractmethod
    def equal(
        self, tester: BaseEqualityTester, object1: T, object2: Any, show_difference: bool = False
    ) -> bool:
        r"""Indicates if two objects are equal or not.

        Args:
            tester (``BaseEqualityTester``): Specifies an equality
                tester.
            object1: Specifies the first object to compare.
            object2: Specifies the second object to compare.
            show_difference (bool, optional): If ``True``, it shows
                a difference between the two objects if they are
                different. This parameter is useful to find the
                difference between two objects. Default: ``False``

        Returns:
            bool: ``True`` if the two objects are equal, otherwise
                ``False``.
        """

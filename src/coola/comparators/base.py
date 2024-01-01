r"""Define the equality operator base classes."""

from __future__ import annotations

__all__ = ["BaseAllCloseOperator", "BaseEqualityOperator"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from coola.testers.base import BaseAllCloseTester, BaseEqualityTester

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseAllCloseOperator(ABC, Generic[T]):
    r"""Defines the base class to implement an equality operator.

    Example usage:

    ```pycon
    >>> from coola.testers import AllCloseTester
    >>> from coola.comparators import DefaultAllCloseOperator
    >>> tester = AllCloseTester()
    >>> op = DefaultAllCloseOperator()
    >>> op.allclose(tester, 42, 42)
    True

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @abstractmethod
    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: T,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        r"""Indicate if two objects are equal within a tolerance.

        Args:
            tester: Specifies an equality tester.
            object1: Specifies the first object to compare.
            object2: Specifies the second object to compare.
            rtol: Specifies the relative tolerance parameter.
            atol: Specifies the absolute tolerance  parameter.
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
        >>> from coola.testers import AllCloseTester
        >>> from coola.comparators import DefaultAllCloseOperator
        >>> tester = AllCloseTester()
        >>> op = DefaultAllCloseOperator()
        >>> op.allclose(tester, 42, 42)
        True
        >>> op.allclose(tester, "meow", "meov")
        False

        ```
        """

    @abstractmethod
    def clone(self) -> BaseAllCloseOperator:
        r"""Return a copy of the equality operator.

        Returns:
            A copy of the equality operator.

        Example usage:

        ```pycon
        >>> from coola.comparators import DefaultAllCloseOperator
        >>> op = DefaultAllCloseOperator()
        >>> op_cloned = op.clone()
        >>> op_cloned
        DefaultAllCloseOperator()
        >>> op is op_cloned
        False

        ```
        """


class BaseEqualityOperator(ABC, Generic[T]):
    r"""Defines the base class to implement an equality operator.

    Example usage:

    ```pycon
    >>> from coola.comparators import DefaultEqualityOperator
    >>> from coola.testers import EqualityTester
    >>> tester = EqualityTester()
    >>> op = DefaultEqualityOperator()
    >>> op.equal(tester, 42, 42)
    True

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @abstractmethod
    def clone(self) -> BaseEqualityOperator:
        r"""Return a copy of the equality operator.

        Returns:
            A copy of the equality operator.

        Example usage:

        ```pycon
        >>> from coola.comparators import DefaultEqualityOperator
        >>> op = DefaultEqualityOperator()
        >>> op_cloned = op.clone()
        >>> op_cloned
        DefaultEqualityOperator()
        >>> op is op_cloned
        False

        ```
        """

    @abstractmethod
    def equal(
        self, tester: BaseEqualityTester, object1: T, object2: Any, show_difference: bool = False
    ) -> bool:
        r"""Indicate if two objects are equal or not.

        Args:
            tester: Specifies an equality tester.
            object1: Specifies the first object to compare.
            object2: Specifies the second object to compare.
            show_difference: If ``True``, it shows a difference between
                the two objects if they are different. This parameter
                is useful to find the difference between two objects.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.comparators import DefaultEqualityOperator
        >>> from coola.testers import EqualityTester
        >>> tester = EqualityTester()
        >>> op = DefaultEqualityOperator()
        >>> op.equal(tester, 42, 42)
        True
        >>> op.equal(tester, "meow", "meov")
        False

        ```
        """

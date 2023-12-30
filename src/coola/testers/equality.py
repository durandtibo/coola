from __future__ import annotations

__all__ = ["EqualityTester", "LocalEqualityTester"]

import logging
from typing import TYPE_CHECKING, Any

from coola.testers.base import BaseEqualityTester
from coola.utils.format import str_indent, str_mapping

if TYPE_CHECKING:
    from coola.comparators.base import BaseEqualityOperator

logger = logging.getLogger(__name__)


class EqualityTester(BaseEqualityTester):
    """Implements the default equality tester."""

    registry: dict[type[object], BaseEqualityOperator] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_operator(
        cls, data_type: type[object], operator: BaseEqualityOperator, exist_ok: bool = False
    ) -> None:
        r"""Adds an equality operator for a given data type.

        Args:
            data_type: Specifies the data type for this test.
            operator: Specifies the operator used to test the equality
                of the specified type.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                data type already exists. This parameter should be set
                to ``True`` to overwrite the operator for a type.

        Raises:
            RuntimeError: if an operator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from coola.testers import EqualityTester
        >>> from coola.comparators import SequenceEqualityOperator
        >>> EqualityTester.add_operator(list, SequenceEqualityOperator(), exist_ok=True)

        ```
        """
        if data_type in cls.registry and not exist_ok:
            msg = (
                f"An operator ({cls.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "operator for this type"
            )
            raise RuntimeError(msg)
        cls.registry[data_type] = operator

    def equal(self, object1: Any, object2: Any, show_difference: bool = False) -> bool:
        return self.find_operator(type(object1)).equal(self, object1, object2, show_difference)

    @classmethod
    def has_operator(cls, data_type: type[object]) -> bool:
        r"""Indicates if an equality operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            ``True`` if an equality operator is registered,
                otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.testers import EqualityTester
        >>> EqualityTester.has_operator(list)
        True
        >>> EqualityTester.has_operator(str)
        False

        ```
        """
        return data_type in cls.registry

    @classmethod
    def find_operator(cls, data_type: Any) -> BaseEqualityOperator:
        r"""Finds the equality operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            The equality operator associated to the data type.

        Example usage:

        ```pycon
        >>> from coola.testers import EqualityTester
        >>> EqualityTester.find_operator(list)
        SequenceEqualityOperator()
        >>> EqualityTester.find_operator(str)
        DefaultEqualityOperator()

        ```
        """
        for object_type in data_type.__mro__:
            operator = cls.registry.get(object_type, None)
            if operator is not None:
                return operator
        msg = f"Incorrect data type: {data_type}"
        raise TypeError(msg)

    @classmethod
    def local_copy(cls) -> LocalEqualityTester:
        r"""Returns a copy of ``EqualityTester`` that can easily be
        customized without changind ``EqualityTester``.

        Returns:
            A "local" copy of ``EqualityTester``.

        Example usage:

        ```pycon
        >>> from coola.testers import EqualityTester
        >>> tester = EqualityTester.local_copy()
        >>> tester
        LocalEqualityTester(...)

        ```
        """
        return LocalEqualityTester({key: value.clone() for key, value in cls.registry.items()})


class LocalEqualityTester(BaseEqualityTester):
    """Implements an equality tester that can be easily customized.

    Args:
        registry: Specifies the initial registry with the equality
            operators.
    """

    def __init__(self, registry: dict[type[object], BaseEqualityOperator] | None = None) -> None:
        self.registry = registry or {}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.registry == other.registry

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    def add_operator(
        self, data_type: type[object], operator: BaseEqualityOperator, exist_ok: bool = False
    ) -> None:
        r"""Adds an equality operator for a given data type.

        Args:
            data_type: Specifies the data type for this test.
            operator: Specifies the operator used to test the equality
                of the specified type.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                data type already exists. This parameter should be
                set to ``True`` to overwrite the operator for a type.

        Raises:
            RuntimeError: if an operator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from coola.testers import EqualityTester
        >>> from coola.comparators import DefaultEqualityOperator
        >>> tester = EqualityTester.local_copy()
        >>> tester.add_operator(str, DefaultEqualityOperator())
        >>> tester.add_operator(str, DefaultEqualityOperator(), exist_ok=True)

        ```
        """
        if data_type in self.registry and not exist_ok:
            msg = (
                f"An operator ({self.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "operator for this type"
            )
            raise RuntimeError(msg)
        self.registry[data_type] = operator

    def clone(self) -> LocalEqualityTester:
        r"""Clones the current tester.

         Returns:
             A deep copy of the current tester.

         Example usage:

         ```pycon
         >>> import torch
         >>> from coola.testers import EqualityTester
         >>> tester = EqualityTester.local_copy()
         >>> tester_cloned = tester.clone()

        ```
        """
        return self.__class__({key: value.clone() for key, value in self.registry.items()})

    def equal(self, object1: Any, object2: Any, show_difference: bool = False) -> bool:
        return self.find_operator(type(object1)).equal(self, object1, object2, show_difference)

    def has_operator(self, data_type: type[object]) -> bool:
        r"""Indicates if an equality operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            ``True`` if an equality operator is registered,
                otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.testers import EqualityTester
        >>> tester = EqualityTester.local_copy()
        >>> tester.has_operator(list)
        True
        >>> tester.has_operator(str)
        False

        ```
        """
        return data_type in self.registry

    def find_operator(self, data_type: Any) -> BaseEqualityOperator:
        r"""Finds the equality operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            The equality operator associated to the data type.

        Example usage:

        ```pycon
        >>> from coola.testers import EqualityTester
        >>> tester = EqualityTester.local_copy()
        >>> tester.find_operator(list)
        SequenceEqualityOperator()
        >>> tester.find_operator(str)
        DefaultEqualityOperator()

        ```
        """
        for object_type in data_type.__mro__:
            operator = self.registry.get(object_type, None)
            if operator is not None:
                return operator
        msg = f"Incorrect data type: {data_type}"
        raise TypeError(msg)

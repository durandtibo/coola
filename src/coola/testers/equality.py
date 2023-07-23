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
            operator (``BaseEqualityOperator``): Specifies the operator
                used to test the equality of the specified type.
            exist_ok (bool, optional): If ``False``, ``RuntimeError``
                is raised if the data type already exists. This
                parameter should be set to ``True`` to overwrite the
                operator for a type. Default: ``False``.

        Raises:
            RuntimeError if an operator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import EqualityTester
            >>> from coola.comparators import SequenceEqualityOperator
            >>> EqualityTester.add_operator(list, SequenceEqualityOperator(), exist_ok=True)
        """
        if data_type in cls.registry and not exist_ok:
            raise RuntimeError(
                f"An operator ({cls.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "operator for this type"
            )
        cls.registry[data_type] = operator

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
            >>> from coola import EqualityTester
            >>> tester = EqualityTester()
            >>> tester.equal(
            ...     [torch.ones(2, 3), torch.zeros(2)],
            ...     [torch.ones(2, 3), torch.zeros(2)],
            ... )
            True
            >>> tester.equal([torch.ones(2, 3), torch.ones(2)], [torch.ones(2, 3), torch.zeros(2)])
            False
        """
        return self.find_operator(type(object1)).equal(self, object1, object2, show_difference)

    @classmethod
    def has_operator(cls, data_type: type[object]) -> bool:
        r"""Indicates if an equality operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            bool: ``True`` if an equality operator is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import EqualityTester
            >>> EqualityTester.has_operator(list)
            True
            >>> EqualityTester.has_operator(str)
            False
        """
        return data_type in cls.registry

    @classmethod
    def find_operator(cls, data_type: Any) -> BaseEqualityOperator:
        r"""Finds the equality operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            ``BaseEqualityOperator``: The equality operator associated
                to the data type.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import EqualityTester
            >>> EqualityTester.find_operator(list)
            SequenceEqualityOperator()
            >>> EqualityTester.find_operator(str)
            DefaultEqualityOperator()
        """
        for object_type in data_type.__mro__:
            operator = cls.registry.get(object_type, None)
            if operator is not None:
                return operator
        raise TypeError(f"Incorrect data type: {data_type}")

    @classmethod
    def local_copy(cls) -> LocalEqualityTester:
        r"""Returns a copy of ``EqualityTester`` that can easily be
        customized without changind ``EqualityTester``.

        Returns:
            ``LocalEqualityTester``: A local copy of
                ``EqualityTester``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import EqualityTester
            >>> tester = EqualityTester.local_copy()
            >>> tester  # doctest: +ELLIPSIS
            LocalEqualityTester(...)
        """
        return LocalEqualityTester({key: value.clone() for key, value in cls.registry.items()})


class LocalEqualityTester(BaseEqualityTester):
    """Implements an equality tester that can be easily customized.

    Args:
        registry (dict or ``None``, optional): Specifies the initial
            registry with the equality operators. Default: ``None``
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
            operator (``BaseEqualityOperator``): Specifies the operator
                used to test the equality of the specified type.
            exist_ok (bool, optional): If ``False``, ``RuntimeError``
                is raised if the data type already exists. This
                parameter should be set to ``True`` to overwrite the
                operator for a type. Default: ``False``.

        Raises:
            RuntimeError if an operator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import EqualityTester
            >>> from coola.comparators import DefaultEqualityOperator
            >>> tester = EqualityTester.local_copy()
            >>> tester.add_operator(str, DefaultEqualityOperator())
            >>> tester.add_operator(str, DefaultEqualityOperator(), exist_ok=True)
        """
        if data_type in self.registry and not exist_ok:
            raise RuntimeError(
                f"An operator ({self.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "operator for this type"
            )
        self.registry[data_type] = operator

    def clone(self) -> LocalEqualityTester:
        r"""Clones the current tester.

        Returns:
            ``LocalEqualityTester``: A deep copy of the current
                tester.

        Example usage:

        .. code-block:: pycon

           >>> import torch
           >>> from coola.testers import EqualityTester
           >>> tester = EqualityTester.local_copy()
           >>> tester_cloned = tester.clone()
        """
        return self.__class__({key: value.clone() for key, value in self.registry.items()})

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
            >>> from coola.testers import EqualityTester
            >>> tester = EqualityTester.local_copy()
            >>> tester.equal(
            ...     [torch.ones(2, 3), torch.zeros(2)],
            ...     [torch.ones(2, 3), torch.zeros(2)],
            ... )
            True
            >>> tester.equal([torch.ones(2, 3), torch.ones(2)], [torch.ones(2, 3), torch.zeros(2)])
            False
        """
        return self.find_operator(type(object1)).equal(self, object1, object2, show_difference)

    def has_operator(self, data_type: type[object]) -> bool:
        r"""Indicates if an equality operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            bool: ``True`` if an equality operator is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import EqualityTester
            >>> tester = EqualityTester.local_copy()
            >>> tester.has_operator(list)
            True
            >>> tester.has_operator(str)
            False
        """
        return data_type in self.registry

    def find_operator(self, data_type: Any) -> BaseEqualityOperator:
        r"""Finds the equality operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            ``BaseEqualityOperator``: The equality operator associated
                to the data type.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import EqualityTester
            >>> tester = EqualityTester.local_copy()
            >>> tester.find_operator(list)
            SequenceEqualityOperator()
            >>> tester.find_operator(str)
            DefaultEqualityOperator()
        """
        for object_type in data_type.__mro__:
            operator = self.registry.get(object_type, None)
            if operator is not None:
                return operator
        raise TypeError(f"Incorrect data type: {data_type}")

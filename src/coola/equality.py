__all__ = [
    "BaseEqualityOperator",
    "BaseEqualityTester",
    "DefaultEqualityOperator",
    "EqualityTester",
    "MappingEqualityOperator",
    "SequenceEqualityOperator",
    "objects_are_equal",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Optional, Type, TypeVar

from coola.format import str_dict, str_indent

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

        .. code-block:: python

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


def objects_are_equal(
    object1: Any,
    object2: Any,
    show_difference: bool = False,
    tester: Optional[BaseEqualityTester] = None,
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

    .. code-block:: python

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


class BaseEqualityOperator(ABC, Generic[T]):
    r"""Define the base class to implement an equality operator."""

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

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


class DefaultEqualityOperator(BaseEqualityOperator[Any]):
    r"""Implements a default equality operator.

    The ``==`` operator is used to test the equality between the
    objects.
    """

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Any,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if not type(object1) is type(object2):
            if show_difference:
                logger.info(f"Objects have different types: {type(object1)} vs {type(object2)}")
            return False
        object_equal = object1 == object2
        if show_difference and not object_equal:
            logger.info(f"Objects are different:\nobject1={object1}\nobject2={object2}")
        return object_equal


class MappingEqualityOperator(BaseEqualityOperator[Mapping]):
    r"""Implements an equality operator for mappings."""

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Mapping,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if not type(object1) is type(object2):
            if show_difference:
                logger.info(
                    f"The mappings have different types: {type(object1)} vs {type(object2)}"
                )
            return False
        if len(object1) != len(object2):
            if show_difference:
                logger.info(f"The mappings have different sizes: {len(object1)} vs {len(object2)}")
            return False
        if set(object1.keys()) != set(object2.keys()):
            if show_difference:
                logger.info(
                    f"The mappings have different keys:\n"
                    f"keys of object1: {sorted(set(object1.keys()))}\n"
                    f"keys of object2: {sorted(set(object2.keys()))}"
                )
            return False
        for key in object1.keys():
            if not tester.equal(object1[key], object2[key], show_difference):
                if show_difference:
                    logger.info(
                        f"The mappings have a different value for the key '{key}':\n"
                        f"first mapping  = {object1}\n"
                        f"second mapping = {object2}"
                    )
                return False
        return True


class SequenceEqualityOperator(BaseEqualityOperator[Sequence]):
    r"""Implements an equality operator for sequences."""

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Sequence,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if not type(object1) is type(object2):
            if show_difference:
                logger.info(
                    f"The sequences have different types: {type(object1)} vs {type(object2)}"
                )
            return False
        if len(object1) != len(object2):
            if show_difference:
                logger.info(
                    f"The sequences have different sizes: {len(object1):,} vs {len(object2):,}"
                )
            return False
        for value1, value2 in zip(object1, object2):
            if not tester.equal(value1, value2, show_difference):
                if show_difference:
                    logger.info(
                        f"The sequences have at least one different value:\n"
                        f"first sequence  = {object1}\n"
                        f"second sequence = {object2}"
                    )
                return False
        return True


class EqualityTester(BaseEqualityTester):
    """Implements the default equality tester."""

    registry: dict[Type[object], BaseEqualityOperator] = {
        Mapping: MappingEqualityOperator(),
        Sequence: SequenceEqualityOperator(),
        dict: MappingEqualityOperator(),
        list: SequenceEqualityOperator(),
        object: DefaultEqualityOperator(),
        tuple: SequenceEqualityOperator(),
    }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n  "
            f"{str_indent(str_dict({str(key): value for key, value in self.registry.items()}))}\n)"
        )

    @classmethod
    def add_equality_operator(
        cls, data_type: Type[object], operator: BaseEqualityOperator, exist_ok: bool = False
    ) -> None:
        r"""Adds an equality operator for a given data type.

        Args:
            data_type: Specifies the data type for this test.
            operator (``BaseEqualityOperator``): Specifies the operator
                used to test the equality of the specified type.
            exist_ok (bool, optional): If ``False``, ``ValueError`` is
                raised if the data type already exists. This parameter
                should be set to ``True`` to overwrite the operator for
                a type. Default: ``False``.

        Raises:
            ValueError if an operator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        .. code-block:: python

            >>> from coola import EqualityTester, BaseEqualityTester, BaseEqualityOperator
            >>> class MyStringEqualityOperator(BaseEqualityOperator[str]):
            ...    def equal(
            ...         self,
            ...         tester: BaseEqualityTester,
            ...         object1: str,
            ...         object2: Any,
            ...         show_difference: bool = False,
            ...     ) -> bool:
            ...         ...  # Custom implementation to test strings
            >>> EqualityTester.add_equality_operator(str, MyStringEqualityOperator())
            # To overwrite an existing operato
            >>> EqualityTester.add_equality_operator(str, MyStringEqualityOperator(), exist_ok=True)
        """
        if data_type in cls.registry and not exist_ok:
            raise ValueError(
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

        .. code-block:: python

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
        return self.find_equality_operator(type(object1)).equal(
            self, object1, object2, show_difference
        )

    @classmethod
    def has_equality_operator(cls, data_type: Type[object]) -> bool:
        r"""Indicates if an equality operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            bool: ``True`` if an equality operator is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: python

            >>> from coola import EqualityTester
            >>> EqualityTester.has_equality_operator(list)
            True
            >>> EqualityTester.has_equality_operator(str)
            False
        """
        return data_type in cls.registry

    @classmethod
    def find_equality_operator(cls, data_type: Any) -> BaseEqualityOperator:
        r"""Finds the equality operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            ``BaseEqualityOperator``: The equality operator associated
                to the data type.

        Example usage:

        .. code-block:: python

            >>> from coola import EqualityTester
            >>> EqualityTester.find_equality_operator(list)
            SequenceEqualityOperator()
            >>> EqualityTester.find_equality_operator(str)
            DefaultEqualityOperator()
        """
        for object_type in data_type.__mro__:
            operator = cls.registry.get(object_type, None)
            if operator is not None:
                return operator
        raise TypeError(f"Incorrect data type: {data_type}")

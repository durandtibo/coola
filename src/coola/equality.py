from __future__ import annotations

__all__ = [
    "BaseEqualityOperator",
    "BaseEqualityTester",
    "EqualityTester",
    "LocalEqualityTester",
    "objects_are_equal",
]

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

from coola.utils.format import str_dict, str_indent

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


class DefaultEqualityOperator(BaseEqualityOperator[Any]):
    r"""Implements a default equality operator.

    The ``==`` operator is used to test the equality between the
    objects.
    """

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> DefaultEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Any,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if type(object1) is not type(object2):
            if show_difference:
                logger.info(f"Objects have different types: {type(object1)} vs {type(object2)}")
            return False
        object_equal = object1 == object2
        if show_difference and not object_equal:
            logger.info(f"Objects are different:\nobject1={object1}\nobject2={object2}")
        return object_equal


class MappingEqualityOperator(BaseEqualityOperator[Mapping]):
    r"""Implements an equality operator for mappings."""

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> MappingEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Mapping,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if type(object1) is not type(object2):
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

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> SequenceEqualityOperator:
        return self.__class__()

    def equal(
        self,
        tester: BaseEqualityTester,
        object1: Sequence,
        object2: Any,
        show_difference: bool = False,
    ) -> bool:
        if object1 is object2:
            return True
        if type(object1) is not type(object2):
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

    registry: dict[type[object], BaseEqualityOperator] = {
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

            >>> from coola import EqualityTester, BaseEqualityTester, BaseEqualityOperator
            >>> class MyStringEqualityOperator(BaseEqualityOperator[str]):
            ...     def equal(
            ...         self,
            ...         tester: BaseEqualityTester,
            ...         object1: str,
            ...         object2: Any,
            ...         show_difference: bool = False,
            ...     ) -> bool:
            ...         ...  # Custom implementation to test strings
            ...
            >>> EqualityTester.add_operator(str, MyStringEqualityOperator())
            # To overwrite an existing operator
            >>> EqualityTester.add_operator(str, MyStringEqualityOperator(), exist_ok=True)
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

            >>> from coola import EqualityTester
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

            >>> from coola import EqualityTester
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
        r"""Returns a copy of ``EqualityTester`` that can
        easily be customized without changind ``EqualityTester``.

        Returns:
            ``LocalEqualityTester``: A local copy of
                ``EqualityTester``.

        Example usage:

        .. code-block:: pycon

            >>> from coola import EqualityTester
            >>> EqualityTester.local_copy()
            LocalEqualityTester(
              <class 'collections.abc.Mapping'>           : MappingEqualityOperator()
              <class 'collections.abc.Sequence'>          : SequenceEqualityOperator()
              <class 'dict'>                              : MappingEqualityOperator()
              <class 'list'>                              : SequenceEqualityOperator()
              <class 'object'>                            : DefaultEqualityOperator()
              <class 'tuple'>                             : SequenceEqualityOperator()
              <class 'numpy.ndarray'>                     : NDArrayEqualityOperator(check_dtype=True)
              <class 'torch.nn.utils.rnn.PackedSequence'> : PackedSequenceEqualityOperator()
              <class 'torch.Tensor'>                      : TensorEqualityOperator()
            )
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
        return (
            f"{self.__class__.__qualname__}(\n  "
            f"{str_indent(str_dict({str(key): value for key, value in self.registry.items()}))}\n)"
        )

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

            >>> from coola import LocalEqualityTester, BaseEqualityTester, BaseEqualityOperator
            >>> class MyStringEqualityOperator(BaseEqualityOperator[str]):
            ...     def equal(
            ...         self,
            ...         tester: BaseEqualityTester,
            ...         object1: str,
            ...         object2: Any,
            ...         show_difference: bool = False,
            ...     ) -> bool:
            ...         ...  # Custom implementation to test strings
            ...
            >>> tester = LocalEqualityTester({...})
            >>> tester.add_operator(str, MyStringEqualityOperator())
            # To overwrite an existing operator
            >>> tester.add_operator(str, MyStringEqualityOperator(), exist_ok=True)
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
            >>> from coola import LocalEqualityTester
            >>> tester = LocalEqualityTester({...})
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
            >>> from coola import LocalEqualityTester
            >>> tester = LocalEqualityTester({...})
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

            >>> from coola import LocalEqualityTester
            >>> tester = LocalEqualityTester({...})
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

            >>> from coola import LocalEqualityTester
            >>> tester = LocalEqualityTester({...})
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

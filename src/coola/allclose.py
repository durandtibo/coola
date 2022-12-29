__all__ = [
    "AllCloseTester",
    "BaseAllCloseOperator",
    "BaseAllCloseTester",
    "DefaultAllCloseOperator",
    "MappingAllCloseOperator",
    "ScalarAllCloseOperator",
    "SequenceAllCloseOperator",
    "objects_are_allclose",
]

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Optional, Type, TypeVar, Union

from coola.format import str_dict, str_indent

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseAllCloseTester(ABC):
    r"""Defines the base class to implement an allclose tester."""

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
                difference between the two objects if they are
                different. This parameter is useful to find the
                difference between two objects. Default: ``False``

        Returns:
            bool: ``True`` if the two objects are equal within a
                tolerance, otherwise ``False``

        Example usage:

        .. code-block:: python

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
        """


def objects_are_allclose(
    object1: Any,
    object2: Any,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
    show_difference: bool = False,
    tester: Optional[BaseAllCloseTester] = None,
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

    .. code-block:: python

        >>> import torch
        >>> from coola import objects_are_allclose
        >>> objects_are_allclose(
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ...     [torch.ones(2, 3), torch.zeros(2)],
        ...     )
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


class BaseAllCloseOperator(ABC, Generic[T]):
    r"""Define the base class to implement an equality operator."""

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
        r"""Indicates if two objects are equal within a tolerance.

        Args:
            tester (``BaseAllCloseTester``): Specifies an equality
                tester.
            object1: Specifies the first object to compare.
            object2: Specifies the second object to compare.
            rtol (float, optional): Specifies the relative tolerance
                parameter. Default: ``1e-5``
            atol (float, optional): Specifies the absolute tolerance
                parameter. Default: ``1e-8``
            equal_nan (bool, optional): If ``True``, then two ``NaN``s
                will be considered equal. Default: ``False``
            show_difference (bool, optional): If ``True``, it shows a
                difference between the two objects if they are
                different. This parameter is useful to find the
                difference between two objects. Default: ``False``

        Returns:
            bool: ``True`` if the two objects are equal within a
                tolerance, otherwise ``False``
        """


class DefaultAllCloseOperator(BaseAllCloseOperator[Any]):
    r"""Implements a default allclose operator.

    The ``==`` operator is used to test the equality between the objects
    because it is not possible to define an allclose operator for all
    objects.
    """

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Any,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
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


class MappingAllCloseOperator(BaseAllCloseOperator[Mapping]):
    r"""Implements an equality operator for mappings."""

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Mapping,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
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
            if not tester.allclose(
                object1[key], object2[key], rtol, atol, equal_nan, show_difference
            ):
                if show_difference:
                    logger.info(
                        f"The mappings have a different value for the key {key}:\n"
                        f"first mapping  = {object1}\n"
                        f"second mapping = {object2}"
                    )
                return False
        return True


class ScalarAllCloseOperator(BaseAllCloseOperator[Union[bool, int, float]]):
    r"""Implements an allclose operator for scalar values."""

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Union[bool, int, float],
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
    ) -> bool:
        if not type(object1) is type(object2):
            if show_difference:
                logger.info(f"Objects have different types: {type(object1)} vs {type(object2)}")
            return False
        number_equal = math.isclose(object1, object2, rel_tol=rtol, abs_tol=atol)
        if show_difference and not number_equal:
            logger.info(f"The numbers are different: {object1} vs {object2}")
        return number_equal


class SequenceAllCloseOperator(BaseAllCloseOperator[Sequence]):
    r"""Implements an equality operator for sequences."""

    def allclose(
        self,
        tester: BaseAllCloseTester,
        object1: Sequence,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
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
            if not tester.allclose(value1, value2, rtol, atol, equal_nan, show_difference):
                if show_difference:
                    logger.info(
                        f"The sequences have at least one different value:\n"
                        f"first sequence  = {object1}\n"
                        f"second sequence = {object2}"
                    )
                return False
        return True


class AllCloseTester(BaseAllCloseTester):
    """Implements the default allclose tester.

    By default, this tester uses the following mapping to test the
    objects:

        - ``PackedSequence``: ``PackedSequenceAllCloseOperator``
        - ``Tensor``: ``TensorAllCloseOperator``
        - ``bool``: ``ScalarAllCloseOperator``
        - ``dict``: ``MappingAllCloseOperator``
        - ``float``: ``ScalarAllCloseOperator``
        - ``int``: ``ScalarAllCloseOperator``
        - ``list``: ``SequenceAllCloseOperator``
        - ``np.ndarray``: ``NDArrayAllCloseOperator``
        - ``object``: ``DefaultAllCloseOperator``
        - ``tuple``: ``SequenceAllCloseOperator``
    """

    registry: dict[Type[object], BaseAllCloseOperator] = {
        Mapping: MappingAllCloseOperator(),
        Sequence: SequenceAllCloseOperator(),
        bool: ScalarAllCloseOperator(),
        dict: MappingAllCloseOperator(),
        float: ScalarAllCloseOperator(),
        int: ScalarAllCloseOperator(),
        list: SequenceAllCloseOperator(),
        object: DefaultAllCloseOperator(),
        tuple: SequenceAllCloseOperator(),
    }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(\n  "
            f"{str_indent(str_dict({str(key): value for key, value in self.registry.items()}))}\n)"
        )

    @classmethod
    def add_allclose_operator(
        cls, data_type: Type[object], operator: BaseAllCloseOperator, exist_ok: bool = False
    ) -> None:
        r"""Adds an allclose operator for a given data type.

        Args:
            data_type: Specifies the data type for this test.
            operator (``BaseAllCloseOperator``): Specifies the operator
                used to test the allclose equality of the specified
                type.
            exist_ok (bool, optional): If ``False``, ``ValueError`` is
                raised if the data type already exists. This parameter
                should be set to ``True`` to overwrite the operator for
                a type. Default: ``False``.

        Raises:
            ValueError if an operator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        .. code-block:: python

            >>> from coola import (
            ...     AllCloseTester,
            ...     BaseAllCloseTester,
            ...     BaseAllCloseOperator,
            ... )
            >>> class MyStringAllCloseOperator(BaseAllCloseOperator[str]):
            ...    def allclose(
            ...         self,
            ...         tester: BaseAllCloseTester,
            ...         object1: str,
            ...         object2: Any,
            ...         rtol: float = 1e-5,
            ...         atol: float = 1e-8,
            ...         equal_nan: bool = False,
            ...         show_difference: bool = False,
            ...     ) -> bool:
            ...         ...  # Custom implementation to test strings
            >>> AllCloseTester.add_allclose_operator(str, MyStringAllCloseOperator())
            # To overwrite an existing operato
            >>> AllCloseTester.add_allclose_operator(str, MyStringAllCloseOperator(), exist_ok=True)
        """
        if data_type in cls.registry and not exist_ok:
            raise ValueError(
                f"An operator ({cls.registry[data_type]}) is already registered for the data "
                f"type {data_type}.Please use `exist_ok=True` if you want to overwrite the "
                "operator for this type"
            )
        cls.registry[data_type] = operator

    def allclose(
        self,
        object1: Any,
        object2: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        equal_nan: bool = False,
        show_difference: bool = False,
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
                difference between the two objects if they are
                different. This parameter is useful to find the
                difference between two objects. Default: ``False``

        Returns:
            bool: ``True`` if the two objects are equal within a
                tolerance, otherwise ``False``

        Example usage:

        .. code-block:: python

            >>> import torch
            >>> from coola import AllCloseTester
            >>> tester = AllCloseTester()
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
        """
        return self.find_allclose_operator(type(object1)).allclose(
            self, object1, object2, rtol, atol, equal_nan, show_difference
        )

    @classmethod
    def has_allclose_operator(cls, data_type: Type[object]) -> bool:
        r"""Indicates if an allclose operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            bool: ``True`` if an allclose operator is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: python

            >>> from coola import AllCloseTester
            >>> AllCloseTester.has_allclose_operator(list)
            True
            >>> AllCloseTester.has_allclose_operator(str)
            False
        """
        return data_type in cls.registry

    @classmethod
    def find_allclose_operator(cls, data_type: Any) -> BaseAllCloseOperator:
        r"""Finds the allclose operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            ``BaseAllCloseOperator``: The allclose operator associated
                to the data type.

        Example usage:

        .. code-block:: python

            >>> from coola import AllCloseTester
            >>> AllCloseTester.find_allclose_operator(list)
            SequenceAllCloseOperator()
            >>> AllCloseTester.has_allclose_operator(str)
            DefaultAllCloseOperator()
        """
        for object_type in data_type.__mro__:
            operator = cls.registry.get(object_type, None)
            if operator is not None:
                return operator
        raise TypeError(f"Incorrect data type: {data_type}")

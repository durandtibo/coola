from __future__ import annotations

__all__ = ["AllCloseTester", "LocalAllCloseTester"]

import logging
from typing import TYPE_CHECKING, Any

from coola.testers.base import BaseAllCloseTester
from coola.utils.format import str_indent, str_mapping

if TYPE_CHECKING:
    from coola.comparators.base import BaseAllCloseOperator

logger = logging.getLogger(__name__)


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

    registry: dict[type[object], BaseAllCloseOperator] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_operator(
        cls, data_type: type[object], operator: BaseAllCloseOperator, exist_ok: bool = False
    ) -> None:
        r"""Adds an allclose operator for a given data type.

        Args:
            data_type: Specifies the data type for this test.
            operator (``BaseAllCloseOperator``): Specifies the operator
                used to test the allclose equality of the specified
                type.
            exist_ok (bool, optional): If ``False``, ``RuntimeError``
                is raised if the data type already exists. This
                parameter should be set to ``True`` to overwrite the
                operator for a type. Default: ``False``.

        Raises:
            RuntimeError if an operator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import AllCloseTester
            >>> from coola.comparators import SequenceAllCloseOperator
            >>> AllCloseTester.add_operator(list, SequenceAllCloseOperator(), exist_ok=True)
        """
        if data_type in cls.registry and not exist_ok:
            raise RuntimeError(
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

        .. code-block:: pycon

            >>> import torch
            >>> from coola.testers import AllCloseTester
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
        return self.find_operator(type(object1)).allclose(
            self, object1, object2, rtol, atol, equal_nan, show_difference
        )

    @classmethod
    def has_operator(cls, data_type: type[object]) -> bool:
        r"""Indicates if an allclose operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            bool: ``True`` if an allclose operator is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import AllCloseTester
            >>> AllCloseTester.has_operator(list)
            True
            >>> AllCloseTester.has_operator(str)
            False
        """
        return data_type in cls.registry

    @classmethod
    def find_operator(cls, data_type: Any) -> BaseAllCloseOperator:
        r"""Finds the allclose operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            ``BaseAllCloseOperator``: The allclose operator associated
                to the data type.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import AllCloseTester
            >>> AllCloseTester.find_operator(list)
            SequenceAllCloseOperator()
            >>> AllCloseTester.find_operator(str)
            DefaultAllCloseOperator()
        """
        for object_type in data_type.__mro__:
            operator = cls.registry.get(object_type, None)
            if operator is not None:
                return operator
        raise TypeError(f"Incorrect data type: {data_type}")

    @classmethod
    def local_copy(cls) -> LocalAllCloseTester:
        r"""Returns a copy of ``AllCloseTester`` that can easily be
        customized without changind ``AllCloseTester``.

        Returns:
            ``LocalAllCloseTester``: A local copy of
                ``AllCloseTester``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import AllCloseTester
            >>> AllCloseTester.local_copy()  # doctest: +ELLIPSIS
            LocalAllCloseTester(...)
        """
        return LocalAllCloseTester({key: value.clone() for key, value in cls.registry.items()})


class LocalAllCloseTester(BaseAllCloseTester):
    """Implements an equality tester that can be easily customized.

    Args:
        registry (dict or ``None``, optional): Specifies the initial
            registry with the equality operators. Default: ``None``
    """

    def __init__(self, registry: dict[type[object], BaseAllCloseOperator] | None = None) -> None:
        self.registry = registry or {}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.registry == other.registry

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    def add_operator(
        self, data_type: type[object], operator: BaseAllCloseOperator, exist_ok: bool = False
    ) -> None:
        r"""Adds an allclose operator for a given data type.

        Args:
            data_type: Specifies the data type for this test.
            operator (``BaseAllCloseOperator``): Specifies the operator
                used to test the allclose equality of the specified
                type.
            exist_ok (bool, optional): If ``False``, ``RuntimeError``
                is raised if the data type already exists. This
                parameter should be set to ``True`` to overwrite the
                operator for a type. Default: ``False``.

        Raises:
            RuntimeError if an operator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import AllCloseTester
            >>> from coola.comparators import SequenceAllCloseOperator
            >>> tester = AllCloseTester.local_copy()
            >>> tester.add_operator(list, SequenceAllCloseOperator(), exist_ok=True)
        """
        if data_type in self.registry and not exist_ok:
            raise RuntimeError(
                f"An operator ({self.registry[data_type]}) is already registered for the data "
                f"type {data_type}.Please use `exist_ok=True` if you want to overwrite the "
                "operator for this type"
            )
        self.registry[data_type] = operator

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

        .. code-block:: pycon

            >>> import torch
            >>> from coola.testers import AllCloseTester
            >>> tester = AllCloseTester.local_copy()
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
        return self.find_operator(type(object1)).allclose(
            self, object1, object2, rtol, atol, equal_nan, show_difference
        )

    def clone(self) -> LocalAllCloseTester:
        r"""Clones the current tester.

        Returns:
            ``LocalAllCloseTester``: A deep copy of the current
                tester.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import AllCloseTester
            >>> tester = AllCloseTester.local_copy()
            >>> tester_cloned = tester.clone()
        """
        return self.__class__({key: value.clone() for key, value in self.registry.items()})

    def has_operator(self, data_type: type[object]) -> bool:
        r"""Indicates if an allclose operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            bool: ``True`` if an allclose operator is registered,
                otherwise ``False``.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import AllCloseTester
            >>> tester = AllCloseTester.local_copy()
            >>> tester.has_operator(list)
            True
            >>> tester.has_operator(str)
            False
        """
        return data_type in self.registry

    def find_operator(self, data_type: Any) -> BaseAllCloseOperator:
        r"""Finds the allclose operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            ``BaseAllCloseOperator``: The allclose operator associated
                to the data type.

        Example usage:

        .. code-block:: pycon

            >>> from coola.testers import AllCloseTester
            >>> tester = AllCloseTester.local_copy()
            >>> tester.find_operator(list)
            SequenceAllCloseOperator()
            >>> tester.find_operator(str)
            DefaultAllCloseOperator()
        """
        for object_type in data_type.__mro__:
            operator = self.registry.get(object_type, None)
            if operator is not None:
                return operator
        raise TypeError(f"Incorrect data type: {data_type}")

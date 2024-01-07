r"""Implement the default equality tester."""

from __future__ import annotations

__all__ = ["EqualityTester", "LocalEqualityTester"]

import logging
from typing import TYPE_CHECKING, Any

from coola.equality.testers.base import BaseEqualityTester
from coola.utils.format import str_indent, str_mapping

if TYPE_CHECKING:
    from coola.equality import EqualityConfig
    from coola.equality.comparators.base import BaseEqualityComparator

logger = logging.getLogger(__name__)


class EqualityTester(BaseEqualityTester):
    """Implement the default equality tester."""

    registry: dict[type, BaseEqualityComparator] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_operator(
        cls, data_type: type, operator: BaseEqualityComparator, exist_ok: bool = False
    ) -> None:
        r"""Add an equality operator for a given data type.

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
        >>> from coola.equality.testers import EqualityTester
        >>> from coola.equality.comparators import SequenceEqualityComparator
        >>> EqualityTester.add_operator(list, SequenceEqualityComparator(), exist_ok=True)

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

    def equal(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        return self.find_operator(type(object1)).equal(object1, object2, config)

    @classmethod
    def has_operator(cls, data_type: type) -> bool:
        r"""Indicate if an equality operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            ``True`` if an equality operator is registered,
                otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> EqualityTester.has_operator(list)
        True
        >>> EqualityTester.has_operator(str)
        False

        ```
        """
        return data_type in cls.registry

    @classmethod
    def find_operator(cls, data_type: Any) -> BaseEqualityComparator:
        r"""Find the equality operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            The equality operator associated to the data type.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> EqualityTester.find_operator(list)
        SequenceEqualityComparator()
        >>> EqualityTester.find_operator(str)
        DefaultEqualityComparator()

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
        r"""Return a copy of ``EqualityTester`` that can easily be
        customized without changind ``EqualityTester``.

        Returns:
            A "local" copy of ``EqualityTester``.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> tester = EqualityTester.local_copy()
        >>> tester
        LocalEqualityTester(...)

        ```
        """
        return LocalEqualityTester({key: value.clone() for key, value in cls.registry.items()})


class LocalEqualityTester(BaseEqualityTester):
    """Implement an equality tester that can be easily customized.

    Args:
        registry: Specifies the initial registry with the equality
            operators.
    """

    def __init__(self, registry: dict[type, BaseEqualityComparator] | None = None) -> None:
        self.registry = registry or {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.registry == other.registry

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    def add_operator(
        self, data_type: type, operator: BaseEqualityComparator, exist_ok: bool = False
    ) -> None:
        r"""Add an equality operator for a given data type.

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
        >>> from coola.equality.testers import EqualityTester
        >>> from coola.equality.comparators import DefaultEqualityComparator
        >>> tester = EqualityTester.local_copy()
        >>> tester.add_operator(str, DefaultEqualityComparator())
        >>> tester.add_operator(str, DefaultEqualityComparator(), exist_ok=True)

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
         >>> from coola.equality.testers import EqualityTester
         >>> tester = EqualityTester.local_copy()
         >>> tester_cloned = tester.clone()

        ```
        """
        return self.__class__({key: value.clone() for key, value in self.registry.items()})

    def equal(self, object1: Any, object2: Any, config: EqualityConfig) -> bool:
        return self.find_operator(type(object1)).equal(object1, object2, config)

    def has_operator(self, data_type: type) -> bool:
        r"""Indicate if an equality operator is registered for the given
        data type.

        Args:
            data_type: Specifies the data type to check.

        Returns:
            ``True`` if an equality operator is registered,
                otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> tester = EqualityTester.local_copy()
        >>> tester.has_operator(list)
        True
        >>> tester.has_operator(str)
        False

        ```
        """
        return data_type in self.registry

    def find_operator(self, data_type: Any) -> BaseEqualityComparator:
        r"""Find the equality operator associated to an object.

        Args:
            data_type: Specifies the data type to get.

        Returns:
            The equality operator associated to the data type.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> tester = EqualityTester.local_copy()
        >>> tester.find_operator(list)
        SequenceEqualityComparator()
        >>> tester.find_operator(str)
        DefaultEqualityComparator()

        ```
        """
        for object_type in data_type.__mro__:
            operator = self.registry.get(object_type, None)
            if operator is not None:
                return operator
        msg = f"Incorrect data type: {data_type}"
        raise TypeError(msg)

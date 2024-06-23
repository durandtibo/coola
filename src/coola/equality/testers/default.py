r"""Contain the implementation of the default equality tester."""

from __future__ import annotations

__all__ = ["EqualityTester", "LocalEqualityTester"]

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from coola.equality.testers.base import BaseEqualityTester
from coola.utils.format import str_indent, str_mapping

if TYPE_CHECKING:
    from coola.equality import EqualityConfig
    from coola.equality.comparators.base import BaseEqualityComparator

logger = logging.getLogger(__name__)


class EqualityTester(BaseEqualityTester):
    """Implement the default equality tester."""

    registry: ClassVar[dict[type, BaseEqualityComparator]] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    @classmethod
    def add_comparator(
        cls, data_type: type, comparator: BaseEqualityComparator, exist_ok: bool = False
    ) -> None:
        r"""Add an equality comparator for a given data type.

        Args:
            data_type: The data type for this test.
            comparator: The comparator used to test the equality
                of the specified type.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                data type already exists. This parameter should be set
                to ``True`` to overwrite the comparator for a type.

        Raises:
            RuntimeError: if a comparator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> from coola.equality.comparators import SequenceEqualityComparator
        >>> EqualityTester.add_comparator(list, SequenceEqualityComparator(), exist_ok=True)

        ```
        """
        if data_type in cls.registry and not exist_ok:
            msg = (
                f"An comparator ({cls.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "comparator for this type"
            )
            raise RuntimeError(msg)
        cls.registry[data_type] = comparator

    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        return self.find_comparator(type(actual)).equal(actual, expected, config)

    @classmethod
    def has_comparator(cls, data_type: type) -> bool:
        r"""Indicate if an equality comparator is registered for the
        given data type.

        Args:
            data_type: The data type to check.

        Returns:
            ``True`` if an equality comparator is registered,
                otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> EqualityTester.has_comparator(list)
        True
        >>> EqualityTester.has_comparator(str)
        False

        ```
        """
        return data_type in cls.registry

    @classmethod
    def find_comparator(cls, data_type: Any) -> BaseEqualityComparator:
        r"""Find the equality comparator associated to an object.

        Args:
            data_type: The data type to get.

        Returns:
            The equality comparator associated to the data type.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> EqualityTester.find_comparator(list)
        SequenceEqualityComparator()
        >>> EqualityTester.find_comparator(str)
        DefaultEqualityComparator()

        ```
        """
        for object_type in data_type.__mro__:
            comparator = cls.registry.get(object_type, None)
            if comparator is not None:
                return comparator
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
        registry: The initial registry with the equality
            comparators.
    """

    def __init__(self, registry: dict[type, BaseEqualityComparator] | None = None) -> None:
        self.registry = registry or {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.registry == other.registry

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self.registry))}\n)"

    def add_comparator(
        self, data_type: type, comparator: BaseEqualityComparator, exist_ok: bool = False
    ) -> None:
        r"""Add an equality comparator for a given data type.

        Args:
            data_type: The data type for this test.
            comparator: The comparator used to test the equality
                of the specified type.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                data type already exists. This parameter should be
                set to ``True`` to overwrite the comparator for a type.

        Raises:
            RuntimeError: if an comparator is already registered for the
                data type and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> from coola.equality.comparators import DefaultEqualityComparator
        >>> tester = EqualityTester.local_copy()
        >>> tester.add_comparator(str, DefaultEqualityComparator())
        >>> tester.add_comparator(str, DefaultEqualityComparator(), exist_ok=True)

        ```
        """
        if data_type in self.registry and not exist_ok:
            msg = (
                f"An comparator ({self.registry[data_type]}) is already registered for the data "
                f"type {data_type}. Please use `exist_ok=True` if you want to overwrite the "
                "comparator for this type"
            )
            raise RuntimeError(msg)
        self.registry[data_type] = comparator

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

    def equal(self, actual: Any, expected: Any, config: EqualityConfig) -> bool:
        return self.find_comparator(type(actual)).equal(actual, expected, config)

    def has_comparator(self, data_type: type) -> bool:
        r"""Indicate if an equality comparator is registered for the
        given data type.

        Args:
            data_type: The data type to check.

        Returns:
            ``True`` if an equality comparator is registered,
                otherwise ``False``.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> tester = EqualityTester.local_copy()
        >>> tester.has_comparator(list)
        True
        >>> tester.has_comparator(str)
        False

        ```
        """
        return data_type in self.registry

    def find_comparator(self, data_type: Any) -> BaseEqualityComparator:
        r"""Find the equality comparator associated to an object.

        Args:
            data_type: The data type to get.

        Returns:
            The equality comparator associated to the data type.

        Example usage:

        ```pycon
        >>> from coola.equality.testers import EqualityTester
        >>> tester = EqualityTester.local_copy()
        >>> tester.find_comparator(list)
        SequenceEqualityComparator()
        >>> tester.find_comparator(str)
        DefaultEqualityComparator()

        ```
        """
        for object_type in data_type.__mro__:
            comparator = self.registry.get(object_type, None)
            if comparator is not None:
                return comparator
        msg = f"Incorrect data type: {data_type}"
        raise TypeError(msg)


def register_equality() -> None:
    r"""Register equality comparators to ``EqualityTester``.

    Example usage:

    ```pycon

    >>> from coola.equality.testers.default import register_equality
    >>> from coola.equality.testers import EqualityTester
    >>> register_equality()
    >>> tester = EqualityTester()
    >>> tester
    EqualityTester(
      (<class 'object'>): DefaultEqualityComparator()
      (<class 'collections.abc.Mapping'>): MappingEqualityComparator()
      (<class 'collections.abc.Sequence'>): SequenceEqualityComparator()
      ...
    )

    ```
    """
    # Local import to avoid cyclic dependency
    from coola.equality.comparators import get_type_comparator_mapping

    for typ, op in get_type_comparator_mapping().items():
        if not EqualityTester.has_comparator(typ):  # pragma: no cover
            EqualityTester.add_comparator(typ, op)


register_equality()

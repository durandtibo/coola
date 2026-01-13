r"""Define the equality tester registry for recursive equality
comparison.

This module provides a registry system that manages and dispatches
equality testers based on data types, enabling recursive equality
checking of nested data structures.
"""

from __future__ import annotations

__all__ = ["EqualityTesterRegistry"]

from typing import TYPE_CHECKING, Any

from coola.equality.tester.base import BaseEqualityTester
from coola.registry import TypeRegistry
from coola.utils.format import repr_indent, repr_mapping, str_indent, str_mapping

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.equality.config import EqualityConfig


class EqualityTesterRegistry:
    """Registry that manages and dispatches equality testers based on
    data type.

    This registry maintains a mapping from Python types to equality tester instances
    and uses the Method Resolution Order (MRO) for type lookup. When checking
    equality, it automatically selects the most specific registered equality tester for
    the data's type, falling back to parent types or a default tester if needed.

    The registry includes an LRU cache for type lookups to optimize performance
    in applications that repeatedly check equality of similar data structures.

    Args:
        initial_state: Optional initial mapping of types to equality testers.
            If provided, the state is copied to prevent external mutations.

    Attributes:
        _state: Internal mapping of registered types to equality testers

    Example:
        Basic usage:

        ```pycon
        >>> from coola.equality.tester import (
        ...     EqualityTesterRegistry,
        ...     SequenceEqualityTester,
        ...     DefaultEqualityTester,
        ... )
        >>> from coola.equality.config import EqualityConfig
        >>> registry = EqualityTesterRegistry(
        ...     {object: DefaultEqualityTester(), list: SequenceEqualityTester()}
        ... )
        >>> registry
        EqualityTesterRegistry(
          (state): TypeRegistry(
              (<class 'object'>): DefaultEqualityTester()
              (<class 'list'>): SequenceEqualityTester()
            )
        )
        >>> config = EqualityConfig()
        >>> registry.objects_are_equal([1, 2, 3], [1, 2, 3], config=config)
        True

        ```
    """

    def __init__(self, initial_state: dict[type, BaseEqualityTester[Any]] | None = None) -> None:
        self._state: TypeRegistry[BaseEqualityTester] = TypeRegistry[BaseEqualityTester](
            initial_state
        )

    def __repr__(self) -> str:
        state = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {state}\n)"

    def __str__(self) -> str:
        state = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {state}\n)"

    def register(
        self,
        data_type: type,
        tester: BaseEqualityTester[Any],
        exist_ok: bool = False,
    ) -> None:
        """Register an equality tester for a given data type.

        This method associates an equality tester instance with a specific Python type.
        When checking equality for data of this type, the registered tester will be used.
        The cache is automatically cleared after registration to ensure consistency.

        Args:
            data_type: The Python type to register (e.g., list, dict, custom classes)
            tester: The equality tester instance that handles this type
            exist_ok: If False (default), raises an error if the type is already
                registered. If True, overwrites the existing registration silently.

        Raises:
            RuntimeError: If the type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.equality.tester import EqualityTesterRegistry, SequenceEqualityTester
            >>> registry = EqualityTesterRegistry()
            >>> registry.register(list, SequenceEqualityTester())
            >>> registry.has_equality_tester(list)
            True

            ```
        """
        self._state.register(data_type, tester, exist_ok=exist_ok)

    def register_many(
        self,
        mapping: Mapping[type, BaseEqualityTester[Any]],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple equality testers at once.

        This is a convenience method for bulk registration that internally calls
        register() for each type-tester pair.

        Args:
            mapping: Dictionary mapping Python types to equality tester instances
            exist_ok: If False (default), raises an error if any type is already
                registered. If True, overwrites existing registrations silently.

        Raises:
            RuntimeError: If any type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.equality.tester import (
            ...     EqualityTesterRegistry,
            ...     SequenceEqualityTester,
            ...     MappingEqualityTester,
            ... )
            >>> registry = EqualityTesterRegistry()
            >>> registry.register_many(
            ...     {
            ...         list: SequenceEqualityTester(),
            ...         dict: MappingEqualityTester(),
            ...     }
            ... )
            >>> registry
            EqualityTesterRegistry(
              (state): TypeRegistry(
                  (<class 'list'>): SequenceEqualityTester()
                  (<class 'dict'>): MappingEqualityTester()
                )
            )

            ```
        """
        self._state.register_many(mapping, exist_ok=exist_ok)

    def has_equality_tester(self, data_type: type) -> bool:
        """Check if an equality tester is explicitly registered for the
        given type.

        Note that this only checks for direct registration. Even if this returns
        False, find_equality_tester() may still return a tester via MRO lookup
        or the default tester.

        Args:
            data_type: The type to check

        Returns:
            True if an equality tester is explicitly registered for this type,
            False otherwise

        Example:
            ```pycon
            >>> from coola.equality.tester import EqualityTesterRegistry, SequenceEqualityTester
            >>> registry = EqualityTesterRegistry()
            >>> registry.register(list, SequenceEqualityTester())
            >>> registry.has_equality_tester(list)
            True
            >>> registry.has_equality_tester(tuple)
            False

            ```
        """
        return data_type in self._state

    def find_equality_tester(self, data_type: type) -> BaseEqualityTester[Any]:
        """Find the appropriate equality tester for a given type.

        Uses the Method Resolution Order (MRO) to find the most specific
        registered equality tester. For example, if you register a tester
        for Sequence but not for list, lists will use the Sequence tester.

        Results are cached using an LRU cache (256 entries) for performance,
        as tester lookup is a hot path in recursive equality checking.

        Args:
            data_type: The Python type to find an equality tester for

        Returns:
            The most specific registered equality tester for this type, a parent
            type's tester via MRO, or the default tester

        Example:
            ```pycon
            >>> from collections.abc import Sequence
            >>> from coola.equality.tester import (
            ...     EqualityTesterRegistry,
            ...     SequenceEqualityTester,
            ...     DefaultEqualityTester,
            ... )
            >>> registry = EqualityTesterRegistry({object: DefaultEqualityTester()})
            >>> registry.register(Sequence, SequenceEqualityTester())
            >>> # list does not inherit from Sequence, so it uses DefaultEqualityTester
            >>> tester = registry.find_equality_tester(list)
            >>> tester
            DefaultEqualityTester()

            ```
        """
        return self._state.resolve(data_type)

    def objects_are_equal(self, actual: object, expected: object, config: EqualityConfig) -> bool:
        """Check if two objects are equal by recursively comparing their
        structure.

        This is the main entry point for equality checking. It automatically:
        1. Determines the actual object's type
        2. Finds the appropriate equality tester
        3. Delegates to that tester's objects_are_equal method
        4. The tester recursively processes nested structures

        Args:
            actual: The actual object.
            expected: The expected object.
            config: The equality configuration.

        Returns:
            True if the objects are equal according to the registered testers,
            False otherwise

        Example:
            Checking if two lists of integers are equal:

            ```pycon
            >>> from coola.equality.config import EqualityConfig
            >>> from coola.equality.tester import get_default_registry
            >>> registry = get_default_registry()
            >>> config = EqualityConfig()
            >>> registry.objects_are_equal([1, 2, 3], [1, 2, 3], config=config)
            True

            ```

            Checking if two lists of tensors are equal:

            ```pycon
            >>> import torch
            >>> from coola.equality.config import EqualityConfig
            >>> from coola.equality.tester import get_default_registry
            >>> registry = get_default_registry()
            >>> config = EqualityConfig()
            >>> registry.objects_are_equal(
            ...     [torch.ones(2, 3), torch.zeros(2)],
            ...     [torch.ones(2, 3), torch.zeros(2)],
            ...     config=config,
            ... )
            True

            ```
        """
        tester = self.find_equality_tester(type(actual))
        return tester.objects_are_equal(actual=actual, expected=expected, config=config)

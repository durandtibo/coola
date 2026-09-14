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
from coola.registry import BaseTypeDispatchRegistry

if TYPE_CHECKING:
    from coola.equality.config import EqualityConfig


class EqualityTesterRegistry(BaseTypeDispatchRegistry[BaseEqualityTester[Any]]):
    """Registry that manages and dispatches equality testers based on
    data type.

    This registry maintains a mapping from Python types to equality tester instances
    and uses the Method Resolution Order (MRO) for type lookup. When checking
    equality, it automatically selects the most specific registered equality tester for
    the data's type, falling back to parent types or a default tester if needed.

    The registry includes an internal cache for type lookups to optimize
    performance in applications that repeatedly check equality of similar data
    structures.

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

    def has_equality_tester(self, data_type: type) -> bool:
        """Type-specific alias for :meth:`has`: check if an equality
        tester is explicitly registered for the given type.

        See :meth:`BaseTypeDispatchRegistry.has` for the full behavior
        description.

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

            ```
        """
        return self.has(data_type)

    def find_equality_tester(self, data_type: type) -> BaseEqualityTester[Any]:
        """Type-specific alias for :meth:`find`: find the appropriate
        equality tester for a given type.

        See :meth:`BaseTypeDispatchRegistry.find` for the full behavior
        description (MRO resolution, caching, and the ``KeyError`` on no
        match).

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
            >>> # Sequence is not in list's MRO, so the registry falls back to DefaultEqualityTester
            >>> tester = registry.find_equality_tester(list)
            >>> tester
            DefaultEqualityTester()

            ```
        """
        return self.find(data_type)

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

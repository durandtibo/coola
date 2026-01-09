r"""Define the manager registry for recursive data transformation.

This module provides a registry system that manages and dispatches
managers based on data types, enabling recursive transformation of
nested data structures while preserving their original structure.
"""

from __future__ import annotations

__all__ = ["RandomManagerRegistry"]

from typing import TYPE_CHECKING, Any

from coola.random.base import BaseRandomManager
from coola.registry import Registry
from coola.utils.format import repr_indent, str_indent

if TYPE_CHECKING:
    from collections.abc import Mapping


class RandomManagerRegistry:
    """Registry that manages and dispatches managers based on data
    type.

    This registry maintains a mapping from Python types to manager instances
    and uses the Method Resolution Order (MRO) for type lookup. When transforming
    data, it automatically selects the most specific registered manager for
    the data's type, falling back to parent types or a default manager if needed.

    The registry includes an LRU cache for type lookups to optimize performance
    in applications that repeatedly transform similar data structures.

    Args:
        initial_state: Optional initial mapping of types to managers.
            If provided, the state is copied to prevent external mutations.

    Attributes:
        _state: Internal mapping of registered types to managers

    Example:
        Basic usage with a random manager:

        ```pycon
        >>> from coola.random import (
        ...     RandomManagerRegistry,
        ...     RandomRandomManager
        ... )
        >>> registry = RandomManagerRegistry(
        ...     {"random": RandomRandomManager()}
        ... )
        >>> registry
        RandomManagerRegistry(
          Registry(
            (random): RandomRandomManager()
          )
        )
        >>> registry.manual_seed(42)

        ```
    """

    def __init__(self, initial_state: dict[str, BaseRandomManager] | None = None) -> None:
        self._state: Registry[str, BaseRandomManager] = Registry[str, BaseRandomManager](
            initial_state
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {repr_indent(self._state)}\n)"

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(self._state)}\n)"

    def register(
            self,
            key: str,
            manager: BaseRandomManager[Any],
            exist_ok: bool = False,
    ) -> None:
        """Register a manager for a given data type.

        This method associates a manager instance with a specific Python type.
        When data of this type is transformed, the registered manager will be used.
        The cache is automatically cleared after registration to ensure consistency.

        Args:
            data_type: The Python type to register (e.g., list, dict, custom classes)
            manager: The manager instance that handles this type
            exist_ok: If False (default), raises an error if the type is already
                registered. If True, overwrites the existing registration silently.

        Raises:
            RuntimeError: If the type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.random import RandomManagerRegistry, SequenceRandomManager
            >>> registry = RandomManagerRegistry()
            >>> registry.register(list, SequenceRandomManager())
            >>> registry.has_random_manager(list)
            True

            ```
        """
        self._state.register(key, manager, exist_ok=exist_ok)

    def register_many(
            self,
            mapping: Mapping[str, BaseRandomManager],
            exist_ok: bool = False,
    ) -> None:
        """Register multiple managers at once.

        This is a convenience method for bulk registration that internally calls
        register() for each type-manager pair.

        Args:
            mapping: Dictionary mapping Python types to manager instances
            exist_ok: If False (default), raises an error if any type is already
                registered. If True, overwrites existing registrations silently.

        Raises:
            RuntimeError: If any type is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.random import (
            ...     RandomManagerRegistry,
            ...     SequenceRandomManager,
            ...     MappingRandomManager,
            ... )
            >>> registry = RandomManagerRegistry()
            >>> registry.register_many(
            ...     {
            ...         list: SequenceRandomManager(),
            ...         dict: MappingRandomManager(),
            ...     }
            ... )
            >>> registry
            RandomManagerRegistry(
              TypeRegistry(
                (<class 'list'>): SequenceRandomManager()
                (<class 'dict'>): MappingRandomManager()
              )
            )

            ```
        """
        self._state.register_many(mapping, exist_ok=exist_ok)

    def has_manager(self, key: str) -> bool:
        """Check if a random manager is explicitly registered for the given
        type.

        Args:
            key: The key to check

        Returns:
            True if a manager is explicitly registered for this type,
            False otherwise

        Example:
            ```pycon
            >>> from coola.random import RandomManagerRegistry, RandomRandomManager
            >>> registry = RandomManagerRegistry()
            >>> registry.register("random", RandomRandomManager())
            >>> registry.has_manager("random")
            True
            >>> registry.has_manager("torch")
            False

            ```
        """
        return key in self._state

    def get_rng_state(self) -> dict[str, Any]:
        return {key: value.get_rng_state() for key, value in self._state.items()}

    def manual_seed(self, seed: int) -> None:
        for value in self._state.values():
            value.manual_seed(seed)

    def set_rng_state(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            self._state[key].set_rng_state(value)

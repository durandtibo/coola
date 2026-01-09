r"""Define the manager registry for random number generator state
management.

This module provides a registry system that manages random number
generator managers, allowing coordinated control of multiple RNG states
across different libraries (e.g., Python's random module, PyTorch,
NumPy).
"""

from __future__ import annotations

__all__ = ["RandomManagerRegistry"]

from typing import TYPE_CHECKING, Any

from coola.random.base import BaseRandomManager
from coola.registry import Registry
from coola.utils.format import repr_indent, str_indent

if TYPE_CHECKING:
    from collections.abc import Mapping


class RandomManagerRegistry(BaseRandomManager):
    """Registry that manages random number generator managers.

    This registry maintains a mapping from string keys to random manager instances,
    enabling centralized control of random number generator states across multiple
    libraries. It provides methods to seed all managers, get and set their states,
    and check for registered managers.

    Args:
        initial_state: Optional initial mapping of string keys to managers.
            If provided, the state is copied to prevent external mutations.

    Attributes:
        _state: Internal registry mapping keys to random managers

    Example:
        Basic usage with a random manager:

        ```pycon
        >>> from coola.random import RandomManagerRegistry, RandomRandomManager
        >>> registry = RandomManagerRegistry({"random": RandomRandomManager()})
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
        manager: BaseRandomManager,
        exist_ok: bool = False,
    ) -> None:
        """Register a random manager with a given key.

        This method associates a manager instance with a specific string key.
        The manager will be used to control the random number generator state
        for its corresponding library.

        Args:
            key: The string key to register (e.g., "random", "torch", "numpy")
            manager: The random manager instance that handles RNG state
            exist_ok: If False (default), raises an error if the key is already
                registered. If True, overwrites the existing registration silently.

        Raises:
            RuntimeError: If the key is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.random import RandomManagerRegistry, RandomRandomManager
            >>> registry = RandomManagerRegistry()
            >>> registry.register("random", RandomRandomManager())
            >>> registry.has_manager("random")
            True

            ```
        """
        self._state.register(key, manager, exist_ok=exist_ok)

    def register_many(
        self,
        mapping: Mapping[str, BaseRandomManager],
        exist_ok: bool = False,
    ) -> None:
        """Register multiple random managers at once.

        This is a convenience method for bulk registration that internally calls
        register() for each key-manager pair.

        Args:
            mapping: Dictionary mapping string keys to random manager instances
            exist_ok: If False (default), raises an error if any key is already
                registered. If True, overwrites existing registrations silently.

        Raises:
            RuntimeError: If any key is already registered and exist_ok is False

        Example:
            ```pycon
            >>> from coola.random import (
            ...     RandomManagerRegistry,
            ...     RandomRandomManager,
            ...     TorchRandomManager,
            ... )
            >>> registry = RandomManagerRegistry()
            >>> registry.register_many(
            ...     {
            ...         "random": RandomRandomManager(),
            ...         "torch": TorchRandomManager(),
            ...     }
            ... )
            >>> registry
            RandomManagerRegistry(
              Registry(
                (random): RandomRandomManager()
                (torch): TorchRandomManager()
              )
            )

            ```
        """
        self._state.register_many(mapping, exist_ok=exist_ok)

    def has_manager(self, key: str) -> bool:
        """Check if a random manager is registered for the given key.

        Args:
            key: The string key to check

        Returns:
            True if a manager is registered for this key, False otherwise

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

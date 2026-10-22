r"""Define the manager registry for random number generator state
management.

This module provides a registry system that manages random number
generator managers, allowing coordinated control of multiple RNG states
across different libraries (e.g., Python's random module, PyTorch,
NumPy).
"""

from __future__ import annotations

__all__ = ["RandomManagerRegistry"]

import logging
from typing import TYPE_CHECKING, Any

from coola.display import MultilineDisplayMixin
from coola.random.base import BaseRandomManager
from coola.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Mapping

logger: logging.Logger = logging.getLogger(__name__)


class RandomManagerRegistry(MultilineDisplayMixin, BaseRandomManager):
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
          (state): Registry(
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

    def _get_repr_kwargs(self) -> dict[str, Any]:
        return {"state": self._state}

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
            AlreadyRegisteredError: If the key is already registered and exist_ok is False

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
            AlreadyRegisteredError: If any key is already registered and exist_ok is False

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
              (state): Registry(
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

    def find_manager(self, key: str) -> BaseRandomManager:
        """Find the random manager registered for the given key.

        Args:
            key: The string key of the manager.

        Returns:
            The random manager registered for ``key``.

        Raises:
            KeyNotRegisteredError: If no manager is registered for ``key``.

        Example:
            ```pycon
            >>> from coola.random import RandomManagerRegistry, RandomRandomManager
            >>> registry = RandomManagerRegistry()
            >>> registry.register("random", RandomRandomManager())
            >>> registry.find_manager("random")
            RandomRandomManager()

            ```
        """
        return self._state[key]

    def unregister(self, key: str) -> BaseRandomManager:
        """Unregister the random manager of the given key.

        The state of the removed manager is no longer saved, restored
        or seeded by the registry.

        Args:
            key: The string key of the manager.

        Returns:
            The random manager that was registered for ``key``.

        Raises:
            KeyNotRegisteredError: If no manager is registered for ``key``.

        Example:
            ```pycon
            >>> from coola.random import RandomManagerRegistry, RandomRandomManager
            >>> registry = RandomManagerRegistry()
            >>> registry.register("random", RandomRandomManager())
            >>> registry.unregister("random")
            RandomRandomManager()
            >>> registry.has_manager("random")
            False

            ```
        """
        return self._state.unregister(key)

    def get_rng_state(self) -> dict[str, Any]:
        return {key: value.get_rng_state() for key, value in self._state.items()}

    def manual_seed(self, seed: int) -> None:
        # Validate once before seeding, so a seed rejected by one backend
        # does not leave the others half-seeded.
        if not isinstance(seed, int) or isinstance(seed, bool):
            msg = f"seed must be an int, got {type(seed)}"
            raise TypeError(msg)
        if not 0 <= seed < 2**63:
            msg = f"seed must be in the range [0, 2**63), got {seed}"
            raise ValueError(msg)
        for value in self._state.values():
            value.manual_seed(seed)

    def set_rng_state(self, state: dict[str, Any]) -> None:
        missing = sorted(key for key in self._state if key not in state)
        if missing:
            logger.warning(
                "No RNG state for the registered key(s) %s; their state is not restored",
                missing,
            )
        for key, value in state.items():
            if key not in self._state:
                logger.warning(
                    "Ignoring RNG state for unregistered key %r; registered keys are %s",
                    key,
                    sorted(self._state.keys()),
                )
                continue
            self._state[key].set_rng_state(value)

r"""Define the manager registry for random number generator state
management.

This module provides a registry system that manages random number
generator managers, allowing coordinated control of multiple RNG states
across different libraries (e.g., Python's random module, PyTorch,
NumPy).
"""

from __future__ import annotations

__all__ = ["RandomManagerRegistry"]

from typing import Any

from coola.random.base import BaseRandomManager
from coola.random.seed import validate_seed
from coola.registry import Registry


class RandomManagerRegistry(Registry[str, BaseRandomManager], BaseRandomManager):
    r"""Implement a registry of random managers that is a manager.

    The registry maps string keys to random managers, in registration
    order. It has the API of ``coola.registry.Registry`` (``register``,
    ``register_many``, ``unregister``, ``copy``, ``in``, ``[]``, ...)
    and it is a ``BaseRandomManager``, so it seeds, saves and restores
    the state of all its managers, and a registry can be registered in
    another registry.

    The state of the registry is a ``dict`` that maps each key to the
    state of its manager.

    Args:
        initial_state: Optional initial mapping of string keys to
            managers. The mapping is copied.

    Example:
        ```pycon
        >>> from coola.random import RandomManagerRegistry, PythonRandomManager
        >>> registry = RandomManagerRegistry({"random": PythonRandomManager()})
        >>> registry
        RandomManagerRegistry(
          (random): PythonRandomManager()
        )
        >>> registry.manual_seed(42)

        ```
    """

    def get_rng_state(self) -> dict[str, Any]:
        r"""Get the state of the random number generators of all the
        managers.

        Returns:
            The state of each manager, by key.
        """
        return {key: manager.get_rng_state() for key, manager in self.items()}

    def manual_seed(self, seed: int) -> None:
        r"""Set the seed of all the random number generators.

        Args:
            seed: The seed.

        Raises:
            TypeError: If the seed is not an ``int``.
            ValueError: If the seed is not in the range ``[0, 2**63)``.
        """
        # Validate once before seeding, so a seed rejected by one backend
        # does not leave the others half-seeded.
        validate_seed(seed)
        for manager in self.values():
            manager.manual_seed(seed)

    def set_rng_state(self, state: dict[str, Any]) -> None:
        r"""Set the RNG state of all the managers.

        The keys of the state must be the registered keys. All the
        managers are restored before an error is raised.

        Args:
            state: The state to restore, as returned by
                ``get_rng_state``.

        Raises:
            ValueError: If the keys of the state differ from the
                registered keys. Nothing is restored.
            Exception: If one manager fails to restore its state, the
                error of this manager is raised. If several managers
                fail, an ``ExceptionGroup`` is raised.

        Example:
            ```pycon
            >>> from coola.random import RandomManagerRegistry, PythonRandomManager
            >>> registry = RandomManagerRegistry({"random": PythonRandomManager()})
            >>> state = registry.get_rng_state()
            >>> registry.set_rng_state(state)

            ```
        """
        if set(state) != set(self.keys()):
            msg = (
                f"The state has the keys {sorted(state)} but the registered keys are "
                f"{sorted(self.keys())}"
            )
            raise ValueError(msg)
        errors: list[Exception] = []
        for key, manager in self.items():
            try:
                manager.set_rng_state(state[key])
            except Exception as exc:  # noqa: BLE001  restore the other managers first
                errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            msg = "Failed to restore the RNG state"
            raise ExceptionGroup(msg, errors)

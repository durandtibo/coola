r"""Define the public interface for managing random number generator
state across different libraries.

This module provides a unified interface for controlling random number
generators (RNGs) from multiple libraries like Python's random module,
NumPy, and PyTorch. It includes functions for setting seeds, getting and
restoring RNG states, and managing a global registry of random managers.
"""

from __future__ import annotations

__all__ = [
    "get_default_registry",
    "get_rng_state",
    "manual_seed",
    "random_seed",
    "register_managers",
    "set_rng_state",
]

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from coola.random.numpy import NumpyRandomManager
from coola.random.random import RandomRandomManager
from coola.random.registry import RandomManagerRegistry
from coola.random.torch import TorchRandomManager
from coola.utils.imports import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from coola.random.base import BaseRandomManager


@contextmanager
def random_seed(
    seed: int, manager: RandomManagerRegistry | None = None
) -> Generator[None, None, None]:
    r"""Implement a context manager to manage the random seed and random
    number generator (RNG) state.

    The context manager sets the specified random seed and
    restores the original RNG state afterward.

    Args:
        seed: The random number generator seed to use while using
            this context manager.
        manager: An optional RandomManagerRegistry instance to use.
            If not provided, the default random manager will be used.

    Example:
        ```pycon
        >>> import numpy
        >>> from coola.random import random_seed
        >>> with random_seed(42):
        ...     print(numpy.random.randn(2, 4))
        ...
        [[...]]
        >>> with random_seed(42):
        ...     print(numpy.random.randn(2, 4))
        ...
        [[...]]

        ```
    """
    manager = manager if manager else get_default_registry()
    state = manager.get_rng_state()
    try:
        manager.manual_seed(seed)
        yield
    finally:
        manager.set_rng_state(state)


def get_rng_state() -> dict[str, Any]:
    r"""Get the current RNG state.

    Returns:
        The current RNG state.

    Example:
        ```pycon
        >>> from coola.random import get_rng_state
        >>> state = get_rng_state()
        >>> state
        {'random': ...}

        ```
    """
    return get_default_registry().get_rng_state()


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers.

    Args:
        seed: The desired random seed.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.random import manual_seed
        >>> manual_seed(42)
        >>> torch.randn(3)
        tensor([...])
        >>> torch.randn(3)
        tensor([...])
        >>> manual_seed(42)
        >>> torch.randn(3)
        tensor([...])

        ```
    """
    get_default_registry().manual_seed(seed)


def set_rng_state(state: dict[str, Any]) -> None:
    r"""Set the RNG state.

    Args:
        state: The new RNG state.

    Example:
        ```pycon
        >>> import torch
        >>> from coola.random import get_rng_state, set_rng_state
        >>> st = get_rng_state()
        >>> set_rng_state(st)

        ```
    """
    get_default_registry().set_rng_state(state)


def register_managers(mapping: Mapping[str, BaseRandomManager], exist_ok: bool = False) -> None:
    """Register custom managers to the default global registry.

    This allows users to add support for custom random number generators
    without modifying global state directly.

    Args:
        mapping: Dictionary mapping manager names to manager instances
        exist_ok: If False, raises error if any manager name already registered

    Example:
        ```pycon
        >>> from coola.random import register_managers, RandomRandomManager
        >>> register_managers({"custom": RandomRandomManager()})  # doctest: +SKIP

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> RandomManagerRegistry:
    """Get or create the default global registry with common random
    managers.

    Returns a singleton registry instance that is pre-configured with managers
    for common random number generation libraries including Python's random module,
    NumPy (if available), and PyTorch (if available).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A RandomManagerRegistry instance with managers registered for:
            - "random": Python's random module (always available)
            - "numpy": NumPy random (if NumPy is installed)
            - "torch": PyTorch random (if PyTorch is installed)

    Notes:
        The singleton pattern means modifications to the returned registry
        affect all future calls to this function. If you need an isolated
        registry, create a new RandomManagerRegistry instance directly.

    Example:
        ```pycon
        >>> from coola.random import get_default_registry
        >>> registry = get_default_registry()
        >>> # Registry is ready to use with available random managers
        >>> registry
        RandomManagerRegistry(
          (state): Registry(
              (random): RandomRandomManager()
              (numpy): NumpyRandomManager()
              (torch): TorchRandomManager()
            )
        )

        ```
    """
    if not hasattr(get_default_registry, "_registry"):
        registry = RandomManagerRegistry()
        _register_default_managers(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


def _register_default_managers(registry: RandomManagerRegistry) -> None:
    """Register default managers for common random number generators.

    This internal function sets up the standard manager mappings
    used by the default registry. It registers managers for Python's
    random module and conditionally registers NumPy and PyTorch managers
    if those libraries are available.

    The registration strategy:
        - "random": RandomRandomManager (always registered)
        - "numpy": NumpyRandomManager (registered if NumPy is available)
        - "torch": TorchRandomManager (registered if PyTorch is available)

    Args:
        registry: The registry to populate with default managers

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    managers: dict[str, BaseRandomManager] = {"random": RandomRandomManager()}
    if is_numpy_available():  # pragma: no cover
        managers["numpy"] = NumpyRandomManager()
    if is_torch_available():  # pragma: no cover
        managers["torch"] = TorchRandomManager()
    registry.register_many(managers)

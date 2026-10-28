r"""Define the public interface for managing random number generator
state across different libraries.

This module provides a unified interface for controlling random number
generators (RNGs) from multiple libraries like Python's random module,
NumPy, and PyTorch. It includes functions for setting seeds, getting and
restoring RNG states, and managing a global registry of random managers.
"""

from __future__ import annotations

__all__ = [
    "get_default_random_manager_registry",
    "get_default_registry",
    "get_rng_state",
    "manual_seed",
    "random_seed",
    "register_managers",
    "set_rng_state",
]

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from coola.random.python import PythonRandomManager
from coola.random.registry import RandomManagerRegistry
from coola.registry.default import make_default_registry_singleton
from coola.utils.imports import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from coola.random.base import BaseRandomManager
    from coola.utils.singleton import LazySingleton


@contextmanager
def random_seed(seed: int, *, registry: RandomManagerRegistry | None = None) -> Generator[None]:
    r"""Temporarily set a random seed and restore previous RNG state.

    Note:
        The first call with the default registry creates it, and it
        imports every installed library that it can seed (NumPy and
        PyTorch), which can take about a second. The libraries are
        imported then, not when they are used later, so that a library
        that is imported after this call is still seeded.

    Args:
        seed: Seed used while inside the context manager.
        registry: Random manager registry to use. If ``None``, the default
            registry is used.

    Example:
        ```pycon
        >>> import random
        >>> from coola.random import random_seed
        >>> with random_seed(42):
        ...     first = random.random()
        ...
        >>> with random_seed(42):
        ...     second = random.random()
        ...
        >>> first == second
        True

        ```
    """
    registry = get_default_registry() if registry is None else registry
    with registry.seed(seed):
        yield


def get_rng_state() -> dict[str, Any]:
    r"""Get the current RNG state.

    Returns:
        Mapping of manager names to their serialized RNG states.

    Example:
        ```pycon
        >>> from coola.random import get_rng_state
        >>> state = get_rng_state()
        >>> "random" in state
        True

        ```
    """
    return get_default_registry().get_rng_state()


def manual_seed(seed: int) -> None:
    r"""Set the seed for generating random numbers.

    Note:
        The first call with the default registry creates it, and it
        imports every installed library that it can seed (NumPy and
        PyTorch), which can take about a second. The libraries are
        imported then, not when they are used later, so that a library
        that is imported after this call is still seeded.

    Args:
        seed: The desired random seed. It must be an ``int`` in the
            range ``[0, 2**63)`` (the range supported by all the
            backends).

    Raises:
        TypeError: If the seed is not an ``int``.
        ValueError: If the seed is out of range. No generator is
            seeded in this case.

    Example:
        ```pycon
        >>> import random
        >>> from coola.random import manual_seed
        >>> manual_seed(42)
        >>> first = random.random()
        >>> manual_seed(42)
        >>> second = random.random()
        >>> first == second
        True

        ```
    """
    get_default_registry().manual_seed(seed)


def set_rng_state(state: dict[str, Any]) -> None:
    r"""Set the RNG state.

    Args:
        state: The new RNG state.

    Example:
        ```pycon
        >>> from coola.random import get_rng_state, set_rng_state
        >>> before = get_rng_state()
        >>> set_rng_state(before)
        >>> after = get_rng_state()
        >>> before.keys() == after.keys()
        True

        ```
    """
    get_default_registry().set_rng_state(state)


def register_managers(mapping: Mapping[str, BaseRandomManager], *, exist_ok: bool = False) -> None:
    """Register custom managers to the default global registry.

    This allows users to add support for custom random number generators
    without modifying global state directly.

    Args:
        mapping: Mapping of manager names to manager instances.
        exist_ok: If ``False``, raise an error when a manager name is
            already registered.

    Example:
        ```pycon
        >>> from coola.random import register_managers, PythonRandomManager
        >>> register_managers({"custom": PythonRandomManager()})  # doctest: +SKIP

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
        A singleton ``RandomManagerRegistry`` with a manager for Python's
        ``random`` module and optional managers for NumPy/PyTorch when those
        dependencies are available.

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
          (random): PythonRandomManager()
          (numpy): NumpyRandomManager()
          (torch): TorchRandomManager()
        )

        ```
    """
    return _default_registry.get()


get_default_random_manager_registry = get_default_registry


def _register_default_managers(registry: RandomManagerRegistry) -> None:
    """Register default managers for common random number generators.

    This internal function sets up the standard manager mappings
    used by the default registry. It registers managers for Python's
    random module and conditionally registers NumPy and PyTorch managers
    if those libraries are available.

    The registration strategy:
        - "random": PythonRandomManager (always registered)
        - "numpy": NumpyRandomManager (registered if NumPy is available)
        - "torch": TorchRandomManager (registered if PyTorch is available)

    Args:
        registry: The registry to populate with default managers

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    registry.register("random", PythonRandomManager())
    # The managers of the optional libraries are imported here, not with ``coola.random``. They
    # must be in the registry before the first seed, because the RNG of a library has to be
    # seeded even if the library is imported later.
    if is_numpy_available():
        from coola.random.numpy import NumpyRandomManager  # noqa: PLC0415

        registry.register("numpy", NumpyRandomManager())
    if is_torch_available():
        from coola.random.torch import TorchRandomManager  # noqa: PLC0415

        registry.register("torch", TorchRandomManager())


_default_registry: LazySingleton[RandomManagerRegistry] = make_default_registry_singleton(
    RandomManagerRegistry, _register_default_managers, plugin_group="coola.random.managers"
)

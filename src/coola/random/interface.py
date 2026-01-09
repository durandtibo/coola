r"""Define the public interface to randomly apply a function to all
items in nested data."""

from __future__ import annotations

__all__ = ["get_default_registry", "register_managers"]

from typing import TYPE_CHECKING

from coola.random.numpy_ import NumpyRandomManager
from coola.random.random_ import RandomRandomManager
from coola.random.registry import RandomManagerRegistry
from coola.random.torch_ import TorchRandomManager
from coola.utils.imports import is_numpy_available, is_torch_available

if TYPE_CHECKING:
    from collections.abc import Mapping

    from coola.random.base import BaseRandomManager


def register_managers(
    mapping: Mapping[str, BaseRandomManager],
    exist_ok: bool = False,
) -> None:
    """Register custom managers to the default global registry.

    This allows users to add support for custom types without modifying
    global state directly.

    Args:
        mapping: Dictionary mapping types to manager instances
        exist_ok: If False, raises error if any type already registered

    Example:
        ```pycon
        >>> from coola.random import register_managers, RandomRandomManager
        >>> register_managers({"default": RandomRandomManager()})

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> RandomManagerRegistry:
    """Get or create the default global registry with common Python
    types.

    Returns a singleton registry instance that is pre-configured with managers
    for Python's built-in types including sequences (list, tuple), mappings (dict),
    sets, and scalar types (int, float, str, bool).

    This function uses a singleton pattern to ensure the same registry instance
    is returned on subsequent calls, which is efficient and maintains consistency
    across an application.

    Returns:
        A RandomManagerRegistry instance with managers registered for:
            - Scalar types (int, float, complex, bool, str)
            - Sequences (list, tuple, Sequence ABC)
            - Sets (set, frozenset)
            - Mappings (dict, Mapping ABC)

    Notes:
        The singleton pattern means modifications to the returned registry
        affect all future calls to this function. If you need an isolated
        registry, create a new RandomManagerRegistry instance directly.

    Example:
        ```pycon
        >>> from coola.random import get_default_registry
        >>> registry = get_default_registry()
        >>> # Registry is ready to use with common Python types
        >>> registry.transform([1, 2, 3], str)
        ['1', '2', '3']
        >>> registry.transform({"a": 1, "b": 2}, lambda x: x * 10)
        {'a': 10, 'b': 20}

        ```
    """
    if not hasattr(get_default_registry, "_registry"):
        registry = RandomManagerRegistry()
        _register_default_managers(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


def _register_default_managers(registry: RandomManagerRegistry) -> None:
    """Register default managers for common Python types.

    This internal function sets up the standard type-to-manager mappings
    used by the default registry. It registers managers in a specific order
    to ensure proper inheritance handling via MRO.

    The registration strategy:
        - Scalar types use DefaultTransformer (no recursion)
        - Strings use DefaultTransformer to prevent character-by-character iteration
        - Sequences use SequenceTransformer for random list/tuple processing
        - Sets use SetTransformer for random set processing
        - Mappings use MappingTransformer for random dict processing

    Args:
        registry: The registry to populate with default managers

    Notes:
        This function is called internally by get_default_registry() and should
        not typically be called directly by users.
    """
    managers: dict[str, BaseRandomManager] = {"random": RandomRandomManager()}
    if is_numpy_available():
        managers["numpy"] = NumpyRandomManager()
    if is_torch_available():
        managers["torch"] = TorchRandomManager()

    registry.register_many(managers)

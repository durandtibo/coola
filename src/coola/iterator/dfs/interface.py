r"""Define the public interface to recursively iterate over nested
data."""

from __future__ import annotations

__all__ = ["dfs_iterate", "get_default_registry", "register_iterators"]

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.default import DefaultIterator
from coola.iterator.dfs.iterable import IterableIterator
from coola.iterator.dfs.mapping import MappingIterator
from coola.iterator.dfs.registry import IteratorRegistry

if TYPE_CHECKING:
    from coola.iterator.dfs.base import BaseIterator


def dfs_iterate(data: Any, registry: IteratorRegistry | None = None) -> Iterator[Any]:
    """Perform Depth-First Search (DFS) iteration over nested data
    structures (lists, dicts, tuples, sets, etc.).

    This function yields elements from the data structure in a DFS manner, recursively
    traversing all levels of nested structures. It uses the appropriate iterators registered
    for the data types (e.g., lists, dictionaries, etc.).

    Args:
        data: The nested data structure to traverse. Can be a list, tuple, dict, set, or any
            other registered type.
        registry: The registry to resolve iterators for nested data. If `None`, the default
            registry is used.

    Yields:
        The elements from the nested data structure in DFS order.

    Example:
        ```pycon
        >>> from coola.iterator import dfs_iterate
        >>> list(dfs_iterate({"a": 1, "b": "abc"}))
        [1, 'abc']
        >>> list(dfs_iterate([1, [2, 3], {"x": 4}]))
        [1, 2, 3, 4]

        ```
    """
    if registry is None:
        registry = get_default_registry()
    yield from registry.iterate(data)


def register_iterators(
    mapping: Mapping[type, BaseIterator[Any]],
    exist_ok: bool = False,
) -> None:
    """Register custom iterators to the default global registry.

    This allows users to add support for custom types without modifying global state directly.

    Args:
        mapping: A dictionary mapping Python types to their corresponding iterator instances.
        exist_ok: If `True`, existing registrations for types will be overwritten.
            If `False`, an error is raised when a type is already registered.

    Example:
        ```pycon
        >>> from coola.iterator.dfs import register_iterators, IterableIterator, IteratorRegistry
        >>> register_iterators({list: IterableIterator()}, exist_ok=True)
        >>> registry = get_default_registry()
        >>> list(registry.iterate([1, 2, 3]))
        [1, 2, 3]

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> IteratorRegistry:
    """Get or create the default global registry for iterators.

    This function returns a singleton instance of the `IteratorRegistry`, which is
    pre-configured with iterators for common Python types, including iterables (lists,
    tuples), mappings (dicts), sets, and scalars (int, float, str, bool). The registry
    is used to look up the appropriate iterator for a given data structure during iteration.

    Returns:
        An `IteratorRegistry` instance with iterators registered for common Python types.

    Notes:
        The singleton pattern means any changes to the returned registry affect all future
        calls to this function. If an isolated registry is needed, create a new `IteratorRegistry`
        instance directly.

    Example:
        ```pycon
        >>> from coola.iterator.dfs import get_default_registry
        >>> reg = get_default_registry()
        >>> list(reg.iterate([1, 2, 3]))
        [1, 2, 3]

        ```
    """
    if not hasattr(get_default_registry, "_registry"):
        registry = IteratorRegistry()
        _register_default_iterators(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


def _register_default_iterators(registry: IteratorRegistry) -> None:
    """Register default iterators for common Python types.

    This internal function registers the standard type-to-iterator mappings that are used
    by the default registry. The registration ensures that each type is handled appropriately
    during iteration, including handling nested structures.

    Args:
        registry: The `IteratorRegistry` to populate with default iterators.

    Notes:
        This function is automatically called by `get_default_registry()` and should not
        be called directly by users.
    """
    default_iterator = DefaultIterator()
    iterable_iterator = IterableIterator()
    mapping_iterator = MappingIterator()

    registry.register_many(
        {
            # Scalar types - no recursion needed
            object: default_iterator,
            str: default_iterator,  # Strings should not be iterated character by character
            bytes: default_iterator,
            int: default_iterator,
            float: default_iterator,
            complex: default_iterator,
            bool: default_iterator,
            # Iterables - recursive iteration (lists, tuples, etc.)
            list: iterable_iterator,
            tuple: iterable_iterator,
            range: iterable_iterator,
            set: iterable_iterator,
            frozenset: iterable_iterator,
            Iterable: iterable_iterator,
            # Mappings - recursive iteration (dictionaries)
            dict: mapping_iterator,
            Mapping: mapping_iterator,
        }
    )

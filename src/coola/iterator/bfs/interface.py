r"""Define the public interface to recursively iterate over nested
data."""

from __future__ import annotations

__all__ = ["bfs_iterate", "get_default_registry", "register_child_finders"]

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.default import DefaultChildFinder
from coola.iterator.bfs.iterable import IterableChildFinder
from coola.iterator.bfs.mapping import MappingChildFinder
from coola.iterator.bfs.registry import ChildFinderRegistry

if TYPE_CHECKING:
    from coola.iterator.bfs import BaseChildFinder


def bfs_iterate(data: Any, registry: ChildFinderRegistry | None = None) -> Iterator[Any]:
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
        Atomic leaf values in BFS order (excludes containers even if empty)

    Example:
        ```pycon
        >>> from coola.iterator import bfs_iterate
        >>> list(bfs_iterate({"a": 1, "b": "abc"}))
        [1, 'abc']
        >>> list(bfs_iterate([1, [2, 3], {"x": 4}]))
        [1, 2, 3, 4]

        ```
    """
    if registry is None:
        registry = get_default_registry()
    yield from registry.iterate(data)


def register_child_finders(
    mapping: Mapping[type, BaseChildFinder[Any]],
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
        >>> from coola.iterator.bfs import (
        ...     register_child_finders,
        ...     IterableChildFinder,
        ...     ChildFinderRegistry,
        ... )
        >>> register_child_finders({list: IterableChildFinder()}, exist_ok=True)
        >>> registry = get_default_registry()
        >>> list(registry.iterate([1, 2, 3]))
        [1, 2, 3]

        ```
    """
    get_default_registry().register_many(mapping, exist_ok=exist_ok)


def get_default_registry() -> ChildFinderRegistry:
    """Get or create the default global registry for iterators.

    This function returns a singleton instance of the `ChildFinderRegistry`, which is
    pre-configured with iterators for common Python types, including iterables (lists,
    tuples), mappings (dicts), sets, and scalars (int, float, str, bool). The registry
    is used to look up the appropriate iterator for a given data structure during iteration.

    Returns:
        An `ChildFinderRegistry` instance with iterators registered for common Python types.

    Notes:
        The singleton pattern means any changes to the returned registry affect all future
        calls to this function. If an isolated registry is needed, create a new `ChildFinderRegistry`
        instance directly.

    Example:
        ```pycon
        >>> from coola.iterator.bfs import get_default_registry
        >>> reg = get_default_registry()
        >>> list(reg.iterate([1, 2, 3]))
        [1, 2, 3]

        ```
    """
    if not hasattr(get_default_registry, "_registry"):
        registry = ChildFinderRegistry()
        _register_default_child_finders(registry)
        get_default_registry._registry = registry
    return get_default_registry._registry


def _register_default_child_finders(registry: ChildFinderRegistry) -> None:
    """Register default iterators for common Python types.

    This internal function registers the standard type-to-iterator mappings that are used
    by the default registry. The registration ensures that each type is handled appropriately
    during iteration, including handling nested structures.

    Args:
        registry: The `ChildFinderRegistry` to populate with default iterators.

    Notes:
        This function is automatically called by `get_default_registry()` and should not
        be called directly by users.
    """
    default_child_finder = DefaultChildFinder()
    iterable_child_finder = IterableChildFinder()
    mapping_child_finder = MappingChildFinder()

    registry.register_many(
        {
            # Scalar types - no recursion needed
            object: default_child_finder,
            str: default_child_finder,  # Strings should not be iterated character by character
            bytes: default_child_finder,
            int: default_child_finder,
            float: default_child_finder,
            complex: default_child_finder,
            bool: default_child_finder,
            # Iterables - recursive iteration (lists, tuples, etc.)
            list: iterable_child_finder,
            tuple: iterable_child_finder,
            range: iterable_child_finder,
            Iterable: iterable_child_finder,
            # Sets - recursive iteration (sets, frozenset)
            set: iterable_child_finder,
            frozenset: iterable_child_finder,
            # Mappings - recursive iteration (dictionaries)
            dict: mapping_child_finder,
            Mapping: mapping_child_finder,
        }
    )

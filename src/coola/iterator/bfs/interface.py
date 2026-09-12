r"""Define the public interface to recursively iterate over nested
data."""

from __future__ import annotations

__all__ = ["bfs_iterate", "get_default_registry", "register_child_finders"]

from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.default import DefaultChildFinder
from coola.iterator.bfs.iterable import IterableChildFinder
from coola.iterator.bfs.mapping import MappingChildFinder
from coola.iterator.bfs.registry import ChildFinderRegistry
from coola.iterator.bootstrap import register_default_handlers
from coola.utils.singleton import LazySingleton

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from coola.iterator.bfs import BaseChildFinder


def bfs_iterate(data: object, registry: ChildFinderRegistry | None = None) -> Iterator[Any]:
    """Perform Breadth-First Search (BFS) iteration over nested data
    structures (lists, dicts, tuples, sets, etc.).

    This function yields elements from the data structure in a BFS manner, recursively
    traversing all levels of nested structures. It uses the appropriate child finders registered
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
    """Register custom child finders to the default global registry.

    This allows users to add support for custom types without modifying global state directly.

    Args:
        mapping: A dictionary mapping Python types to their corresponding child finder instances.
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
    """Get or create the default global registry for child finders.

    This function returns a singleton instance of the `ChildFinderRegistry`, which is
    pre-configured with child finders for common Python types, including iterables (lists,
    tuples), mappings (dicts), sets, and scalars (int, float, str, bool). The registry
    is used to look up the appropriate child finder for a given data structure during iteration.

    Returns:
        An `ChildFinderRegistry` instance with child finders registered for common Python types.

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
    return _default_registry.get()


def _register_default_child_finders(registry: ChildFinderRegistry) -> None:
    """Register default child finders for common Python types.

    This internal function registers the standard type-to-child-finder mappings that are used
    by the default registry. The registration ensures that each type is handled appropriately
    during iteration, including handling nested structures.

    Args:
        registry: The `ChildFinderRegistry` to populate with default child finders.

    Notes:
        This function is automatically called by `get_default_registry()` and should not
        be called directly by users.
    """
    register_default_handlers(
        registry,
        default_handler=DefaultChildFinder(),
        iterable_handler=IterableChildFinder(),
        mapping_handler=MappingChildFinder(),
    )


def _build_default_registry() -> ChildFinderRegistry:
    registry = ChildFinderRegistry()
    _register_default_child_finders(registry)
    return registry


_default_registry: LazySingleton[ChildFinderRegistry] = LazySingleton(_build_default_registry)

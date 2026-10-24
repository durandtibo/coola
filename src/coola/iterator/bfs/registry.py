r"""Define the child finder registry for managing child finders used for
breadth-first traversal of nested data structures."""

from __future__ import annotations

__all__ = ["ChildFinderRegistry"]

from collections import deque
from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.base import BaseChildFinder
from coola.registry import BaseTypeDispatchRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

_Ancestors = tuple[int, "_Ancestors"] | None
r"""A linked list of the ids of the ancestors of an object."""


def _is_ancestor(data: object, ancestors: _Ancestors) -> bool:
    r"""Indicate if ``data`` is one of its ancestors (a cycle)."""
    target = id(data)
    while ancestors is not None:
        ancestor_id, ancestors = ancestors
        if ancestor_id == target:
            return True
    return False


class ChildFinderRegistry(BaseTypeDispatchRegistry[BaseChildFinder[Any]]):
    r"""Registry that manages child finders for breadth-first traversal
    of nested data structures.

    This registry maps Python data types to ``BaseChildFinder`` instances.
    During traversal, the registry selects the most specific child finder
    for a given object using Method Resolution Order (MRO). If no match
    is found, a default child finder is used.

    The registry also caches resolved child finders to speed up repeated
    lookups.

    Args:
        initial_state: An optional dictionary mapping Python types to
            ``BaseChildFinder`` instances. If provided, the registry
            is initialized with this mapping.

    Attributes:
        _state: Mapping of registered data types to child finders.

    Example:
        Basic usage with a flat iterable:

        ```pycon
        >>> from coola.iterator.bfs import (
        ...     ChildFinderRegistry,
        ...     IterableChildFinder,
        ...     DefaultChildFinder,
        ... )
        >>> registry = ChildFinderRegistry(
        ...     {object: DefaultChildFinder(), list: IterableChildFinder()}
        ... )
        >>> list(registry.iterate([1, 2, 3]))
        [1, 2, 3]

        ```

        Working with nested structures using the default registry:

        ```pycon
        >>> from coola.iterator.bfs import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> list(registry.iterate(data))
        [1, 2, 3, 4]

        ```

        Breadth-first traversal over mixed nested data:

        ```pycon
        >>> from coola.iterator.bfs import (
        ...     ChildFinderRegistry,
        ...     IterableChildFinder,
        ...     MappingChildFinder,
        ... )
        >>> registry = ChildFinderRegistry(
        ...     {
        ...         object: DefaultChildFinder(),
        ...         list: IterableChildFinder(),
        ...         dict: MappingChildFinder(),
        ...     }
        ... )
        >>> data = {"a": [1, 2], "b": [3, 4], "c": 5, "d": {"e": 6}}
        >>> list(registry.iterate(data))
        [5, 1, 2, 3, 4, 6]

        ```
    """

    def has_child_finder(self, data_type: type) -> bool:
        r"""Type-specific alias for ``has``: check if a child finder
        is directly registered for a data type.

        See ``BaseTypeDispatchRegistry.has`` for the full behavior
        description.

        Args:
            data_type: The type to check.

        Returns:
            ``True`` if a child finder is directly registered for the
                type, ``False`` otherwise.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import ChildFinderRegistry, IterableChildFinder
            >>> registry = ChildFinderRegistry({list: IterableChildFinder()})
            >>> registry.has_child_finder(list)
            True
            >>> registry.has_child_finder(tuple)
            False

            ```
        """
        return self.has(data_type)

    def find_child_finder(self, data_type: type) -> BaseChildFinder[Any]:
        r"""Type-specific alias for ``find``: find the appropriate
        child finder for a given data type.

        See ``BaseTypeDispatchRegistry.find`` for the full behavior
        description (MRO resolution, caching, and the ``KeyError`` on no
        match).

        Args:
            data_type: The data type for which to find a child finder.

        Returns:
            The resolved child finder instance.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import (
            ...     ChildFinderRegistry,
            ...     IterableChildFinder,
            ...     DefaultChildFinder,
            ... )
            >>> registry = ChildFinderRegistry(
            ...     {object: DefaultChildFinder(), list: IterableChildFinder()}
            ... )
            >>> registry.find_child_finder(list)
            IterableChildFinder()
            >>> registry.find_child_finder(tuple)
            DefaultChildFinder()

            ```
        """
        return self._find_with_dataclass_marker(data_type)

    def find_children(self, data: object) -> Iterator[Any]:
        r"""Return the immediate children of an object using its child
        finder.

        This method does not perform traversal by itself. It delegates
        to the appropriate child finder for the object's type.

        Args:
            data: The object whose children should be extracted.

        Yields:
            Child objects as defined by the resolved child finder.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import (
            ...     ChildFinderRegistry,
            ...     IterableChildFinder,
            ...     DefaultChildFinder,
            ... )
            >>> registry = ChildFinderRegistry(
            ...     {object: DefaultChildFinder(), list: IterableChildFinder()}
            ... )
            >>> list(registry.find_children([1, 2, 3]))
            [1, 2, 3]

            ```
        """
        child_finder = self.find_child_finder(type(data))
        yield from child_finder.find_children(data)

    def iterate(self, data: object) -> Iterator[Any]:
        r"""Perform a breadth-first traversal over a nested data
        structure.

        This method traverses the input data using breadth-first search
        (BFS). Container objects (mappings and iterables, excluding
        strings and bytes) are expanded using registered child finders.
        Only non-container (leaf) values are yielded.

        Containers themselves are never yielded, even if they are empty.
        A value is a container if its child finder is not a leaf
        (``is_leaf``). Iterators and generators are not containers by
        default: they are yielded as they are, never consumed. The value
        of a type whose child finder is not a leaf and finds no child is
        not yielded either, like an empty container.

        Args:
            data: The data structure to traverse.

        Yields:
            Atomic (non-container) values in breadth-first order.

        Raises:
            ValueError: If a container contains itself, directly or
                through other containers (a cycle).

        Example:
            ```pycon
            >>> from coola.iterator.bfs import (
            ...     ChildFinderRegistry,
            ...     IterableChildFinder,
            ...     MappingChildFinder,
            ...     DefaultChildFinder,
            ... )
            >>> registry = ChildFinderRegistry(
            ...     {
            ...         object: DefaultChildFinder(),
            ...         list: IterableChildFinder(),
            ...         dict: MappingChildFinder(),
            ...     }
            ... )
            >>> list(registry.iterate({"a": [1, 2], "b": [3, 4], "c": 5, "d": {"e": 6}}))
            [5, 1, 2, 3, 4, 6]

            ```
        """
        # Each item of the queue comes with its ancestors, as a linked list of
        # ``(id, parent)`` pairs, which is used to detect the cycles.
        queue: deque[tuple[object, _Ancestors]] = deque([(data, None)])
        # A container can be its own ancestor only if it is expanded twice, so the
        # (linear) walk over its ancestors is needed only for the ids already expanded.
        # It keeps a deep chain linear instead of quadratic in its depth.
        expanded: set[int] = set()

        while queue:
            current, ancestors = queue.popleft()
            child_finder = self.find_child_finder(type(current))
            if child_finder.is_leaf:
                yield current
                continue
            if id(current) in expanded and _is_ancestor(current, ancestors):
                msg = f"Cycle detected: an object of type {type(current)} contains itself"
                raise ValueError(msg)
            expanded.add(id(current))
            children_ancestors = (id(current), ancestors)
            queue.extend(
                (child, children_ancestors) for child in child_finder.find_children(current)
            )

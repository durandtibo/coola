r"""Define the child finder registry used by the depth-first and the
breadth-first traversals of nested data."""

from __future__ import annotations

__all__ = ["ChildFinderRegistry"]

from typing import Any

from coola.iterator.base import BaseChildFinder
from coola.registry import BaseTypeDispatchRegistry


class ChildFinderRegistry(BaseTypeDispatchRegistry[BaseChildFinder[Any]]):
    r"""Registry that maps Python types to child finders.

    During a traversal, the registry selects the most specific child
    finder for the type of an object using the Method Resolution Order
    (MRO). The registry is only a dispatcher: the traversal is done by
    ``dfs_iterate`` and ``bfs_iterate``, so one registry serves both
    orders. The resolved child finders are cached.

    Args:
        initial_state: An optional dictionary mapping Python types to
            ``BaseChildFinder`` instances. If provided, the registry
            is initialized with this mapping.

    Attributes:
        _state: Mapping of registered data types to child finders.

    Example:
        ```pycon
        >>> from coola.iterator import (
        ...     ChildFinderRegistry,
        ...     DefaultChildFinder,
        ...     IterableChildFinder,
        ...     bfs_iterate,
        ...     dfs_iterate,
        ... )
        >>> registry = ChildFinderRegistry(
        ...     {object: DefaultChildFinder(), list: IterableChildFinder()}
        ... )
        >>> list(dfs_iterate([1, [2, 3], 4], registry=registry))
        [1, 2, 3, 4]
        >>> list(bfs_iterate([1, [2, 3], 4], registry=registry))
        [1, 4, 2, 3]

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
            >>> from coola.iterator import ChildFinderRegistry, IterableChildFinder
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
        description (MRO resolution, caching, and the error on no
        match).

        Args:
            data_type: The data type for which to find a child finder.

        Returns:
            The resolved child finder instance.

        Raises:
            TypeNotRegisteredError: If no child finder is found.

        Example:
            ```pycon
            >>> from coola.iterator import (
            ...     ChildFinderRegistry,
            ...     DefaultChildFinder,
            ...     IterableChildFinder,
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

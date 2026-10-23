r"""Define the iterator registry for managing iterators based on data
types."""

from __future__ import annotations

__all__ = ["IteratorRegistry", "PathRegistry"]

from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.base import BaseIterator
from coola.registry import BaseTypeDispatchRegistry

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class IteratorRegistry(BaseTypeDispatchRegistry[BaseIterator[Any]]):
    r"""Registry that manages iterators for different data types.

    This registry stores iterators for various data types and handles
    the dispatching of the appropriate iterator based on the data
    type during iteration. It uses Method Resolution Order (MRO) to
    resolve the most specific iterator for a given data type.
    It also supports caching of iterators for performance optimization
    in repetitive iteration tasks.

    Args:
        initial_state: An optional dictionary mapping types to iterators.
            If provided, the registry is initialized with this mapping.

    Attributes:
        _state: Internal mapping of registered data types to iterators.

    Example:
        Basic usage:

        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator, DefaultIterator
        >>> registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
        >>> registry
        IteratorRegistry(
          (state): TypeRegistry(
              (<class 'object'>): DefaultIterator()
              (<class 'list'>): IterableIterator()
            )
        )
        >>> list(registry.iterate([1, 2, 3]))
        [1, 2, 3]

        ```

        Working with nested structures:

        ```pycon
        >>> from coola.iterator.dfs import get_default_registry
        >>> registry = get_default_registry()
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> list(registry.iterate(data))
        [1, 2, 3, 4]

        ```
    """

    def has_iterator(self, data_type: type) -> bool:
        r"""Type-specific alias for ``has``: check if an iterator is
        registered for a given data type.

        See ``BaseTypeDispatchRegistry.has`` for the full behavior
        description.

        Args:
            data_type: The type to check.

        Returns:
            `True` if an iterator is registered for the type, `False` otherwise.

        Example:
            ```pycon
            >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator
            >>> registry = IteratorRegistry({list: IterableIterator()})
            >>> registry.has_iterator(list)
            True
            >>> registry.has_iterator(tuple)
            False

            ```
        """
        return self.has(data_type)

    def find_iterator(self, data_type: type) -> BaseIterator[Any]:
        r"""Type-specific alias for ``find``: find the appropriate
        iterator for a given type.

        See ``BaseTypeDispatchRegistry.find`` for the full behavior
        description (MRO resolution, caching, and the ``KeyError`` on no
        match).

        Args:
            data_type: The data type for which to find an iterator.

        Returns:
            The appropriate iterator for the data type.

        Example:
            ```pycon
            >>> from coola.iterator.dfs import IteratorRegistry, IterableIterator, DefaultIterator
            >>> registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
            >>> registry.find_iterator(list)
            IterableIterator()
            >>> registry.find_iterator(tuple)
            DefaultIterator()

            ```
        """
        return self.find(data_type)

    def iterate(self, data: object) -> Iterator[Any]:
        r"""Perform depth-first iteration over a data structure.

        This method uses the appropriate iterator for the data type, which may be
        retrieved via the registry. The iterator will recursively traverse the data
        structure, yielding elements based on its specific implementation.

        Args:
            data: The data structure to iterate over.

        Yields:
            The elements of the data structure according to the
                appropriate iterator's traversal logic.

        Raises:
            ValueError: If a container contains itself, directly or
                through other containers (a cycle), and its iterator
                visits the children with ``registry.iterate`` (as the
                built-in iterators do).

        Example:
            ```pycon
            >>> from coola.iterator.dfs import (
            ...     IteratorRegistry,
            ...     IterableIterator,
            ...     MappingIterator,
            ...     DefaultIterator,
            ... )
            >>> registry = IteratorRegistry(
            ...     {object: DefaultIterator(), list: IterableIterator(), dict: MappingIterator()}
            ... )
            >>> list(registry.iterate({"a": [1, 2], "b": [3, 4]}))
            [1, 2, 3, 4]

            ```
        """
        yield from PathRegistry(self).iterate(data)


class PathRegistry(IteratorRegistry):
    r"""A view of an iterator registry that knows the path to the object
    being iterated.

    It is the registry given to the iterators by
    ``IteratorRegistry.iterate``, so their recursive calls to
    ``iterate`` can detect a container that contains itself. It shares
    its state with the registry it was created from: the iterators
    registered in one are visible in the other.

    One instance is used for a whole traversal. The ids of the
    containers on the path are kept in a set: an id is added when a
    container is entered and removed when it is left, so checking for a
    cycle costs ``O(1)`` per object, and no object is created per
    visited object.

    Args:
        registry: The registry to view.
        ancestors: The ids of the containers that are already on the
            path to the object to iterate. Defaults to no container.

    Example:
        ```pycon
        >>> from coola.iterator.dfs import (
        ...     DefaultIterator,
        ...     IterableIterator,
        ...     IteratorRegistry,
        ...     PathRegistry,
        ... )
        >>> registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
        >>> data = [1, 2]
        >>> path_registry = PathRegistry(registry, (id(data),))
        >>> list(path_registry.iterate([3, 4]))
        [3, 4]
        >>> list(path_registry.iterate(data))
        Traceback (most recent call last):
            ...
        ValueError: Cycle detected: an object of type <class 'list'> contains itself

        ```
    """

    def __init__(self, registry: IteratorRegistry, ancestors: Iterable[int] = ()) -> None:
        super().__init__()
        # The state is shared on purpose: this registry is a view of ``registry``.
        self._state = registry._state
        self._ancestors: set[int] = set(ancestors)

    def iterate(self, data: object) -> Iterator[Any]:
        r"""Iterate over a data structure that is reached from the path.

        Args:
            data: The data structure to iterate over.

        Yields:
            The elements of the data structure.

        Raises:
            ValueError: If ``data`` is one of the containers of the
                path (a cycle).
        """
        key = id(data)
        if key in self._ancestors:
            msg = f"Cycle detected: an object of type {type(data)} contains itself"
            raise ValueError(msg)
        iterator = self.find_iterator(type(data))
        self._ancestors.add(key)
        try:
            yield from iterator.iterate(data, self)
        finally:
            # Also reached when the generator is closed before the end.
            self._ancestors.discard(key)

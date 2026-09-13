r"""Define the iterator registry for managing iterators based on data
types."""

from __future__ import annotations

__all__ = ["IteratorRegistry"]

from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.base import BaseIterator
from coola.registry import BaseTypeDispatchRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator


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
        r"""Check if an iterator is registered for a given data type.

        This method checks for direct registration. Even if this method returns `False`,
        a suitable iterator might still be found using the MRO lookup.

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
        r"""Find the appropriate iterator for a given type.

        This method uses the MRO to find the most specific iterator. It caches the result
        for performance, so subsequent lookups are faster.

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
        iterator = self.find_iterator(type(data))
        yield from iterator.iterate(data, self)

r"""Define the abstract base class for depth-first search iterators."""

from __future__ import annotations

__all__ = ["BaseIterator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.iterator.dfs.registry import IteratorRegistry

T = TypeVar("T")


class BaseIterator(ABC, Generic[T]):
    r"""Abstract base class for depth-first search iterators.

    This class defines the interface that all DFS iterators must implement.
    Iterators are responsible for traversing specific data types and yielding
    their elements during depth-first search traversal. Custom iterators can
    be registered with an ``IteratorRegistry`` to handle specific data types.

    The generic type parameter ``T`` indicates the type of data this iterator
    is designed to handle, though the ``iterate`` method accepts ``Any`` for
    flexibility.

    Notes:
        - Subclasses must implement the ``iterate`` method.
        - For container types, use ``registry.iterate()`` to recursively
          traverse nested structures.
        - For leaf types, simply yield the data directly.

    Example:
        ```pycon
        >>> from coola.iterator.dfs import IteratorRegistry, DefaultIterator
        >>> iterator = DefaultIterator()
        >>> registry = IteratorRegistry()
        >>> list(iterator.iterate(42, registry))
        [42]
        >>> list(iterator.iterate("hello", registry))
        ['hello']

        ```
    """

    @abstractmethod
    def iterate(self, data: T, registry: IteratorRegistry) -> Iterator[Any]:
        r"""Traverse the data structure and yield elements depth-first.

        This method defines how the iterator traverses its associated data type.
        For container or composite types, it should recursively traverse nested
        elements using ``registry.iterate()`` to delegate to appropriate
        iterators. For leaf types, it should yield the data directly.

        Args:
            data: The data structure to traverse. While typed as ``T`` for
                flexibility, implementations typically expect a specific type
                corresponding to the iterator's purpose.
            registry: The iterator registry used to resolve and dispatch
                iterators for nested data structures. Use ``registry.iterate()``
                to recursively traverse nested elements.

        Yields:
            Elements found during depth-first traversal. The exact type and
            nature of yielded elements depends on the specific iterator
            implementation and traversal strategy.

        Notes:
            - The registry parameter should be used to maintain consistent
              traversal behavior across different data types.
            - Implementations should handle the specific structure of their
              target data type appropriately.

        Example:
            ```pycon
            >>> from coola.iterator.dfs import IteratorRegistry, DefaultIterator
            >>> iterator = DefaultIterator()
            >>> registry = IteratorRegistry()
            >>> list(iterator.iterate(42, registry))
            [42]
            >>> list(iterator.iterate("hello", registry))
            ['hello']

            ```
        """

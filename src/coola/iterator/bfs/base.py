r"""Define the abstract base class for child finders used in breadth-
first search."""

from __future__ import annotations

__all__ = ["BaseChildFinder"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")


class BaseChildFinder(ABC, Generic[T]):
    r"""Abstract base class for child finders used in breadth-first
    search.

    This class defines the interface that all child finders must implement.
    Child finders are responsible for finding and yielding the immediate children
    of a given data structure. Custom child finders can be registered with a
    ``ChildFinderRegistry`` to handle specific data types during BFS traversal.

    The generic type parameter ``T`` indicates the type of data this child finder
    is designed to handle.

    Notes:
        - Subclasses must implement the ``find_children`` method.
        - For leaf types (types with no children), simply return without yielding.

    Example:
        ```pycon
        >>> from coola.iterator.bfs import DefaultChildFinder
        >>> child_finder = DefaultChildFinder()
        >>> list(child_finder.find_children(42))
        []
        >>> list(child_finder.find_children("hello"))
        []

        ```
    """

    @abstractmethod
    def find_children(self, data: T) -> Iterator[Any]:
        r"""Find and yield the immediate children of the given data
        structure.

        This method defines how to extract children from the data structure.
        For container types, this typically means yielding the contained elements.
        For leaf types, this method should return without yielding anything.

        Args:
            data: The data structure whose children should be found.

        Yields:
            The immediate children of the data structure. The type of yielded
            elements depends on the specific data structure being processed.

        Notes:
            - This method should only yield direct children, not recurse deeply.
            - The BFS traversal logic handles visiting children recursively.

        Example:
            ```pycon
            >>> from coola.iterator.bfs import DefaultChildFinder
            >>> child_finder = DefaultChildFinder()
            >>> list(child_finder.find_children(42))
            []
            >>> list(child_finder.find_children("hello"))
            []

            ```
        """

r"""Define the protocol and the abstract base class of the child
finders."""

from __future__ import annotations

__all__ = ["BaseChildFinder", "ChildFinder"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)


class ChildFinder(Protocol[T_contra]):
    r"""Protocol of the child finders.

    A child finder only describes the shape of a type: it returns the
    children of a value. The traversal (order, leaves, cycles, depth)
    is done by ``dfs_iterate`` and ``bfs_iterate``, so the same child
    finder serves both orders.
    """

    def find_children(self, data: T_contra) -> Iterable[Any] | None:
        r"""Find the children of the data.

        Args:
            data: The data whose children should be found.

        Returns:
            ``None`` if ``data`` is a leaf, otherwise the children of
                ``data`` in the order they are visited.
        """


class BaseChildFinder(ABC, Generic[T]):
    r"""Abstract base class for the child finders.

    A child finder describes the shape of the values of a type. It is
    registered in a ``ChildFinderRegistry`` and is used by
    ``dfs_iterate`` and ``bfs_iterate``. The generic type parameter
    ``T`` indicates the type of data this child finder handles.

    Notes:
        - Subclasses must implement the ``find_children`` method.
        - ``find_children`` is pure and cheap: it does not call the
          registry, does not recurse, does not modify the data and
          does not consume the children (it returns them).
        - The result is consumed once, lazily, in order. It can be
          the container itself, a view (``data.values()``) or a
          generator.
        - A leaf returns ``None``: the traversal yields the value
          itself. A container with no child returns an empty
          iterable: it is not a leaf and yields nothing.
        - The children are the values to visit: a child finder never
          returns the keys of a mapping.

    Example:
        ```pycon
        >>> from coola.iterator import BaseChildFinder, dfs_iterate, register_child_finders
        >>> class Node:
        ...     def __init__(self, left, right):
        ...         self.left, self.right = left, right
        ...
        >>> class NodeChildFinder(BaseChildFinder[Node]):
        ...     def find_children(self, data):
        ...         return [data.left, data.right]
        ...
        >>> register_child_finders({Node: NodeChildFinder()})
        >>> list(dfs_iterate(Node(1, Node(2, 3))))
        [1, 2, 3]

        ```
    """

    @abstractmethod
    def find_children(self, data: T) -> Iterable[Any] | None:
        r"""Find the children of the data.

        Args:
            data: The data whose children should be found.

        Returns:
            ``None`` if ``data`` is a leaf, otherwise the children of
                ``data`` in the order they are visited. An empty
                iterable for a container without child.

        Example:
            ```pycon
            >>> from coola.iterator import DefaultChildFinder, IterableChildFinder
            >>> print(DefaultChildFinder().find_children(42))
            None
            >>> IterableChildFinder().find_children([1, 2])
            [1, 2]

            ```
        """

    def find_items(self, data: T) -> Iterable[tuple[Any, Any]] | None:
        r"""Find the labelled children of the data.

        The label of a child is what ``dfs_items`` puts in the path of
        the leaves below it. The default implementation numbers the
        children returned by ``find_children`` (``enumerate``), so a
        child finder that only implements ``find_children`` works with
        ``dfs_items`` and gets integer labels. Override it to give the
        raw key, index or name of each child.

        Args:
            data: The data whose children should be found.

        Returns:
            ``None`` if ``data`` is a leaf, otherwise the pairs
                ``(label, child)`` in the order they are visited.

        Example:
            ```pycon
            >>> from coola.iterator import DefaultChildFinder, MappingChildFinder
            >>> print(DefaultChildFinder().find_items(42))
            None
            >>> list(MappingChildFinder().find_items({"a": 1, "b": 2}))
            [('a', 1), ('b', 2)]

            ```
        """
        children = self.find_children(data)
        if children is None:
            return None
        return enumerate(children)

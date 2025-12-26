r"""Define the default child finder class for BFS traversal."""

from __future__ import annotations

__all__ = ["DefaultChildFinder"]

from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.base import BaseChildFinder

if TYPE_CHECKING:
    from collections.abc import Iterator


class DefaultChildFinder(BaseChildFinder[Any]):
    r"""Default child finder for breadth-first search traversal.

        This child finder serves as the fallback handler for objects that don't have
        a specialized child finder registered. It treats the input data as a leaf node
        with no children, so it yields nothing during traversal.

        The DefaultChildFinder is typically used for:
        - Primitive types (int, float, str, bool, None)
        - Objects without internal structure to traverse
        - Terminal nodes in a data structure

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

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def find_children(self, data: Any) -> Iterator[Any]:  # noqa: ARG002
        return
        yield

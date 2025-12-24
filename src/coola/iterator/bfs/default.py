r"""Define the default BFS iterator class for handling leaf nodes in
traversal."""

from __future__ import annotations

__all__ = ["DefaultChildFinder"]

from typing import TYPE_CHECKING, Any

from coola.iterator.bfs.base import BaseChildFinder

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.iterator.bfs.registry import ChildFinderRegistry


class DefaultChildFinder(BaseChildFinder[Any]):
    r"""Default iterator for breath-first search traversal of leaf nodes.

    This iterator serves as the fallback handler for objects that don't have
    a specialized iterator registered. It treats the input data as a leaf node
    and yields it directly without further traversal.

    The DefaultIterator is typically used for:
    - Primitive types (int, float, str, bool, None)
    - Objects without internal structure to traverse
    - Terminal nodes in a data structure

    Examples:
    ```pycon
    >>> from coola.iterator.bfs import ChildFinderRegistry, DefaultChildFinder
    >>> iterator = DefaultChildFinder()
    >>> registry = ChildFinderRegistry()
    >>> list(iterator.find_children(42, registry))
    []
    >>> list(iterator.find_children("hello", registry))
    []

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def find_children(
        self,
        data: Any,  # noqa: ARG002
        registry: ChildFinderRegistry,  # noqa: ARG002
    ) -> Iterator[Any]:
        return
        yield

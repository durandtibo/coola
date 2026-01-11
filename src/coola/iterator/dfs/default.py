r"""Define the default DFS iterator class for handling leaf nodes in
traversal."""

from __future__ import annotations

__all__ = ["DefaultIterator"]

from typing import TYPE_CHECKING, Any

from coola.iterator.dfs.base import BaseIterator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.iterator.dfs.registry import IteratorRegistry


class DefaultIterator(BaseIterator[Any]):
    r"""Default iterator for depth-first search traversal of leaf nodes.

    This iterator serves as the fallback handler for objects that don't have
    a specialized iterator registered. It treats the input data as a leaf node
    and yields it directly without further traversal.

    The DefaultIterator is typically used for:
    - Primitive types (int, float, str, bool, None)
    - Objects without internal structure to traverse
    - Terminal nodes in a data structure

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

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def iterate(
        self,
        data: Any,
        registry: IteratorRegistry,  # noqa: ARG002
    ) -> Iterator[Any]:
        yield data

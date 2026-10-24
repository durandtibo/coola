r"""Define an iterator class for dataclass instances in depth-first
traversal."""

from __future__ import annotations

__all__ = ["DataclassIterator"]

import dataclasses
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.iterator.dfs.base import BaseIterator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.iterator.dfs.registry import IteratorRegistry


class DataclassIterator(InlineDisplayMixin, BaseIterator[Any]):
    r"""Iterator for performing a depth-first traversal over the fields
    of a dataclass instance.

    The values of the fields are visited in the order of the
    definition of the fields. It is not registered by default: see
    ``coola.iterator.dfs.enter_dataclasses``.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.iterator.dfs import (
        ...     DataclassIterator,
        ...     DefaultIterator,
        ...     IterableIterator,
        ...     IteratorRegistry,
        ... )
        >>> @dataclass
        ... class Config:
        ...     lr: float
        ...     layers: list
        ...
        >>> registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
        >>> list(DataclassIterator().iterate(Config(0.1, [1, 2]), registry))
        [0.1, 1, 2]

        ```
    """

    def iterate(self, data: Any, registry: IteratorRegistry) -> Iterator[Any]:
        for field in dataclasses.fields(data):
            yield from registry.iterate(getattr(data, field.name))

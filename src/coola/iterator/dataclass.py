r"""Define the child finder class for dataclass instances."""

from __future__ import annotations

__all__ = ["DataclassChildFinder"]

import dataclasses
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.iterator.base import BaseChildFinder

if TYPE_CHECKING:
    from collections.abc import Iterator


class DataclassChildFinder(InlineDisplayMixin, BaseChildFinder[Any]):
    r"""Child finder for dataclass instances.

    The children of a dataclass instance are the values of its fields,
    in the order of the definition of the fields. The label of a child
    is the name of its field. It is registered as a
    fallback for ``dataclasses.is_dataclass`` in the default registry. A child finder
    registered for the type of the dataclass has priority; register
    ``DefaultChildFinder`` to keep a dataclass as a leaf.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.iterator import DataclassChildFinder
        >>> @dataclass
        ... class Config:
        ...     lr: float
        ...     layers: list
        ...
        >>> list(DataclassChildFinder().find_children(Config(0.1, [1, 2])))
        [0.1, [1, 2]]

        ```
    """

    def find_children(self, data: Any) -> Iterator[Any]:
        r"""Return the children of an object.

        Args:
            data: The object.

        Returns:
            The values of the fields.
        """
        return (getattr(data, field.name) for field in dataclasses.fields(data))

    def find_items(self, data: Any) -> Iterator[tuple[str, Any]]:
        r"""Return the items of an object, as ``(key, child)`` pairs.

        Args:
            data: The object.

        Returns:
            The ``(field name, value)`` pairs.
        """
        return ((field.name, getattr(data, field.name)) for field in dataclasses.fields(data))

r"""Define the dataclass hasher."""

from __future__ import annotations

__all__ = ["DataclassHasher"]

import dataclasses
from itertools import chain
from typing import Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node


class DataclassHasher(InlineDisplayMixin, BaseHasher[Any]):
    r"""Hasher for dataclass instances.

    The hash combines the qualified class name with a mapping from
    field name to field value, with every value hashed through the
    registry. Only the fields with ``compare=True`` are
    hashed, like in ``objects_are_equal``, so two equal dataclasses have
    the same hash. Because dataclasses have no common base
    class, it cannot be registered for a type: the default registry
    uses it as a fallback (see ``HasherRegistry.register_fallback``)
    for every dataclass that has no hasher of its own.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.hashing import DataclassHasher
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> hasher = DataclassHasher()
        >>> hasher
        DataclassHasher()
        >>> hasher.encode(Point(1, 2)).ordered
        False

        ```
    """

    def encode(self, data: Any) -> Encoding:
        r"""Describe a dataclass instance.

        Args:
            data: The dataclass instance to describe.

        Returns:
            The unordered ``Node`` of the names and the values of the
            fields with ``compare=True``, with the qualified class name
            in the tag.

        Raises:
            TypeError: If ``data`` is not a dataclass instance.
        """
        if not dataclasses.is_dataclass(data) or isinstance(data, type):
            msg = f"Expected a dataclass instance, got {type(data)!r}."
            raise TypeError(msg)
        cls = type(data)
        children = chain.from_iterable(
            (field.name, getattr(data, field.name))
            for field in dataclasses.fields(data)
            if field.compare
        )
        return Node(f"dataclass:{cls.__module__}.{cls.__qualname__}", children, False, 2)

r"""Define the hashers of the objects that are described by their fields
or by their content: the dataclasses and the classes with a
``__coola_hash__`` method."""

from __future__ import annotations

__all__ = ["ContentHasher", "DataclassHasher", "SupportsCoolaHash"]

import dataclasses
from itertools import chain
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class SupportsCoolaHash(Protocol):
    r"""Implement a protocol to represent objects that say what their
    content is.

    The method returns the content, not a hash: the engine hashes it
    through the registry, so the object cannot bypass the protections
    of the engine and does not have to forward any option.

    Example:
        ```pycon
        >>> from coola.hashing import SupportsCoolaHash
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x, self.y = x, y
        ...     def __coola_hash__(self):
        ...         return (self.x, self.y)
        ...
        >>> isinstance(Point(1, 2), SupportsCoolaHash)
        True

        ```
    """

    def __coola_hash__(self) -> object:
        r"""Return the content of this object.

        Returns:
            Any value that can be hashed, for example a tuple of the
            attributes that identify the object.
        """
        ...


class ContentHasher(InlineDisplayMixin, BaseHasher[SupportsCoolaHash]):
    r"""Hasher for objects that implement ``__coola_hash__``.

    The object is described by the tag ``custom:module.Class`` and the
    value of ``__coola_hash__``, which the engine hashes through the
    registry. The qualified class name is part of the tag, so two
    classes with the same content hash differently. The default registry
    uses it as a fallback for every class that defines the method and has
    no hasher of its own.

    Example:
        ```pycon
        >>> from coola.hashing import ContentHasher, hash_object
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x, self.y = x, y
        ...     def __coola_hash__(self):
        ...         return (self.x, self.y)
        ...
        >>> hasher = ContentHasher()
        >>> hasher
        ContentHasher()
        >>> hasher.encode(Point(1, 2)).children
        [(1, 2)]
        >>> hash_object(Point(1, 2)) == hash_object(Point(1, 2))
        True

        ```
    """

    def encode(self, data: SupportsCoolaHash) -> Encoding:
        r"""Describe an object by its content.

        Args:
            data: The object to describe.

        Returns:
            The ``Node`` with the tag ``custom:module.Class`` and the
            value of ``data.__coola_hash__()`` as only child.
        """
        cls = type(data)
        return Node(f"custom:{cls.__module__}.{cls.__qualname__}", [data.__coola_hash__()])

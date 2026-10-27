r"""Define a hasher for objects that say what their content is."""

from __future__ import annotations

__all__ = ["ContentHasher", "SupportsCoolaHash"]

from typing import Protocol, runtime_checkable

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node


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

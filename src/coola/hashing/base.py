r"""Define the hasher base class and the building blocks of the encoding.

A hasher only describes the content of a type: ``encode`` returns the
bytes of a leaf, or a ``Node`` with a tag and the children of a
container. The engine (``coola.hashing.engine``) resolves the hasher of
each child, handles cycles, shared containers and the depth, and
computes the digests.
"""

from __future__ import annotations

__all__ = ["BaseHasher", "Encoding", "Hasher", "Node", "digest", "tagged"]

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)

# The version of the format: it is the ``person`` parameter of BLAKE2b, so the digests of two
# formats can never be mistaken for one another.
PERSON = b"coola-hash-2"


@dataclass(slots=True)
class Node:
    r"""Describe a container: a tag and the children to hash.

    Each child is encoded by its own hasher, found by the registry, so
    the children can be any value, including values built on the spot
    (a list of field names, a NumPy array, a ``str`` dtype).

    Args:
        tag: The constant tag of the hasher.
        children: The values to hash. They are consumed once, lazily.
        ordered: If ``True``, the position of a child matters. If
            ``False``, the groups of children are sorted by digest, so
            the hash does not depend on the iteration order.
        group: The number of consecutive children that form one entry
            of an unordered node (a mapping uses ``2``: key, value).

    Example:
        ```pycon
        >>> from coola.hashing import Node
        >>> node = Node("Point", [1, 2])
        >>> node.tag
        'Point'

        ```
    """

    tag: str
    children: Iterable[Any]
    ordered: bool = True
    group: int = 1


Encoding: TypeAlias = bytes | Node


def tagged(tag: str, payload: str | bytes = b"") -> bytes:
    r"""Return the encoding of a leaf: the tag, a NUL byte and the
    payload.

    The text is encoded in UTF-8 with ``surrogatepass``, so a lone
    surrogate never collides with a valid string.

    Args:
        tag: The constant tag of the type.
        payload: The content of the value.

    Returns:
        The bytes of the leaf.

    Example:
        ```pycon
        >>> from coola.hashing import tagged
        >>> tagged("int", "1")
        b'int\x001'

        ```
    """
    if isinstance(payload, str):
        payload = payload.encode(errors="surrogatepass")
    return tag.encode(errors="surrogatepass") + b"\0" + payload


def digest(data: bytes) -> bytes:
    r"""Return the 64-byte BLAKE2b digest of the package.

    A hasher uses it to reduce a big buffer (the values of an array, the
    content of a ``bytes``) to a fixed-width value before it enters the
    encoding, so the engine never copies a big preimage.

    Args:
        data: The bytes to digest.

    Returns:
        The 64-byte digest.

    Example:
        ```pycon
        >>> from coola.hashing import digest
        >>> len(digest(b"abc"))
        64

        ```
    """
    return hashlib.blake2b(data, digest_size=64, person=PERSON).digest()


class Hasher(Protocol[T_contra]):
    r"""Structural interface of a type-specific hasher.

    Any object with a compatible ``encode`` method can be registered
    without subclassing ``BaseHasher``, which remains available as a
    convenience base class.
    """

    def encode(self, data: T_contra) -> Encoding:
        r"""Describe an object for hashing.

        Args:
            data: The object to describe.

        Returns:
            The ``bytes`` of a leaf, or a ``Node`` for a container.
        """


class BaseHasher(ABC, Generic[T]):
    r"""Base class for type-specific hashers.

    A hasher describes the content of a type, it does not hash. The
    rules of the contract are:

    1. ``encode`` is pure and cheap: it does not call the registry, does
       not recurse, does not modify ``data`` and does not depend on
       ``id()``, on ``hash()`` or on the iteration order of a set or of
       a mapping (an unordered ``Node`` says so instead).
    2. It returns ``bytes`` (a leaf, see ``tagged``) or a ``Node``.
    3. The children of a ``Node`` are encoded by their own hasher.
    4. The tag is a constant of the hasher, except for the user-defined
       classes whose qualified name is the identity. A subclass
       therefore hashes like its base.
    5. An exception propagates unchanged.

    Example:
        ```pycon
        >>> from coola.hashing import BaseHasher, Node, hash_object, register_hashers
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x, self.y = x, y
        ...
        >>> class PointHasher(BaseHasher[Point]):
        ...     def encode(self, data):
        ...         return Node("Point", [data.x, data.y])
        ...
        >>> register_hashers({Point: PointHasher()})
        >>> hash_object(Point(1, 2)) == hash_object(Point(1, 2))
        True
        >>> hash_object(Point(1, 2)) == hash_object(Point(2, 1))
        False

        ```
    """

    @abstractmethod
    def encode(self, data: T) -> Encoding:
        r"""Describe the content of ``data``.

        Args:
            data: The value to describe. Must be of the type ``T`` that
                this hasher handles.

        Returns:
            The bytes of a leaf (``tagged(tag, payload)``), or a
            ``Node`` that gives the tag and the children of a
            container.

        Example:
            ```pycon
            >>> from coola.hashing import StringHasher
            >>> StringHasher().encode("abc")
            b'str\x00abc'

            ```
        """

r"""Provide deterministic hashing functions for strings."""

from __future__ import annotations

__all__ = ["StringHasher", "hash_enum", "hash_string", "hash_tagged"]

import hashlib
from collections import UserString
from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, tagged
from coola.hashing.engine import validate_hash_length

if TYPE_CHECKING:
    from enum import Enum


class StringHasher(InlineDisplayMixin, BaseHasher[str | UserString]):
    r"""Hasher for string objects.

    This hasher describes the string under the ``str`` type tag, without
    any intermediate conversion. ``str`` subclasses hash like plain
    strings (the enum members are dispatched to ``EnumHasher`` by the
    registry). A ``UserString`` is described by its text under the
    ``UserString`` tag, so it differs from a ``str``.

    Example:
        ```pycon
        >>> from coola.hashing import StringHasher
        >>> hasher = StringHasher()
        >>> hasher
        StringHasher()
        >>> hasher.encode("Meowwwwww")
        b'str\x00Meowwwwww'

        ```
    """

    def encode(self, data: str | UserString) -> Encoding:
        r"""Describe a string.

        A lone surrogate (for example in the result of ``os.fsdecode``
        for a file name that is not valid UTF-8) is encoded with the
        ``surrogatepass`` error handler.

        Args:
            data: The string to describe.

        Returns:
            The leaf ``tagged("str", data)``, or
            ``tagged("UserString", str(data))`` for a ``UserString``.
        """
        if isinstance(data, UserString):
            return tagged("UserString", str(data))
        return tagged("str", data)


def hash_string(value: str, length: int = 64) -> str:
    """Generate a hexadecimal hash of a string using BLAKE2b.

    A lone surrogate (for example in the result of ``os.fsdecode`` for a
    file name that is not valid UTF-8) is encoded with the
    ``surrogatepass`` error handler.

    Args:
        value: The input string to hash.
        length: The desired length of the returned hex string. Must be an
            even number between 2 and 128 inclusive, since each byte of the
            digest encodes as two hex characters. Defaults to 64 (32-byte
            digest).

    Returns:
        A lowercase hexadecimal string of exactly ``length`` characters.

    Raises:
        ValueError: If ``length`` is not an even number between 2 and 128.

    Example:
        ```pycon
        >>> from coola.hashing import hash_string
        >>> len(hash_string("hello", length=16))
        16

        ```
    """
    validate_hash_length(length)
    digest_size = length // 2
    # ``surrogatepass`` encodes a lone surrogate (e.g. from ``os.fsdecode``)
    # as three bytes that valid UTF-8 never contains, so it cannot collide
    # with another string and it does not change the other strings.
    encoded = value.encode(errors="surrogatepass")
    return hashlib.blake2b(encoded, digest_size=digest_size).hexdigest()


def hash_tagged(tag: str, value: str, length: int = 64) -> str:
    """Compute a hash of ``value`` namespaced by a type ``tag``.

    The tag is mixed into the digest input so that values of different
    types that have the same textual form (e.g. ``1`` and ``'1'``, or
    an empty list and an empty string) do not collide.

    Args:
        tag: The type tag, e.g. ``"int"`` or ``"list"``.
        value: The text to hash.
        length: The desired length of the returned hex string. See
            ``hash_string`` for constraints.

    Returns:
        A lowercase hexadecimal string of exactly ``length``
        characters.

    Raises:
        ValueError: If ``length`` is not an even number between 2 and
            128.

    Example:
        ```pycon
        >>> from coola.hashing.string import hash_tagged
        >>> hash_tagged("int", "1") == hash_tagged("str", "1")
        False

        ```
    """
    return hash_string(f"{tag}\0{value}", length=length)


def hash_enum(member: Enum, length: int = 64) -> str:
    """Compute a hash of an enum member.

    The hash depends on the qualified name of the enum class and on the
    member name, not on the member value, so it is stable and does not
    collapse ``IntEnum``/``StrEnum`` members into ``int``/``str``.

    Args:
        member: The enum member to hash.
        length: The desired length of the returned hex string. See
            ``hash_string`` for constraints.

    Returns:
        A lowercase hexadecimal string of exactly ``length``
        characters.

    Example:
        ```pycon
        >>> from enum import Enum
        >>> from coola.hashing import hash_enum
        >>> class Color(Enum):
        ...     RED = 1
        ...
        >>> len(hash_enum(Color.RED))
        64

        ```
    """
    cls = type(member)
    name = member.name if member.name is not None else repr(member.value)
    return hash_tagged("Enum", f"{cls.__module__}.{cls.__qualname__}.{name}", length=length)

r"""Provide deterministic hashing functions for strings."""

from __future__ import annotations

__all__ = ["StringHasher", "hash_enum", "hash_string", "hash_tagged"]

import hashlib
from enum import Enum
from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.hashing.base import BaseHasher

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class StringHasher(InlineDisplayMixin, NoArgsDisplayMixin, BaseHasher[str]):
    r"""Hasher for string objects.

    This hasher computes the hash of the string, namespaced by the
    ``str`` type tag, without any intermediate conversion. ``str``
    subclasses hash like plain strings, except enum members, which hash
    through ``hash_enum``.

    Example:
        ```pycon
        >>> from coola.hashing import StringHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = StringHasher()
        >>> hasher
        StringHasher()
        >>> len(hasher.hash("Meowwwwww", registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: str,
        registry: HasherRegistry,  # noqa: ARG002
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        r"""Compute a deterministic hash of a string.

        Args:
            data: The input string to hash.
            registry: The hasher registry. Unused by this hasher since
                strings are hashed directly with no need to dispatch to
                another hasher for nested/composite data; accepted
                only to satisfy the common ``BaseHasher`` interface.
            length: The desired length of the returned hex string. See
                ``hash_string`` for constraints. Defaults to 64.
            ignore_unhashable: Unused by this hasher; accepted only to
                satisfy the common ``BaseHasher`` interface.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        if isinstance(data, Enum):
            # ``(str, Enum)`` mixins reach this hasher through the MRO.
            return hash_enum(data, length=length)
        return hash_tagged("str", data, length=length)


def hash_string(value: str, length: int = 64) -> str:
    """Generate a hexadecimal hash of a string using BLAKE2b.

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
    if length % 2 != 0 or not (2 <= length <= 128):
        msg = f"length must be an even number between 2 and 128, got {length}."
        raise ValueError(msg)
    digest_size = length // 2
    try:
        encoded = value.encode()
    except UnicodeEncodeError as exc:
        msg = f"Cannot hash a string that is not valid UTF-8 (e.g. a lone surrogate): {exc.reason}."
        raise ValueError(msg) from exc
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

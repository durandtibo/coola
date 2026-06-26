r"""Provide deterministic hashing functions for strings."""

from __future__ import annotations

__all__ = ["StringHasher", "hash_string"]

import hashlib
from typing import TYPE_CHECKING

from coola.hashing.base import BaseHasher

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class StringHasher(BaseHasher[str]):
    r"""Hasher for string objects.

    This hasher computes the hash of the string directly, without any
    intermediate conversion.

    Example:
        ```pycon
        >>> from coola.hashing import StringHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = StringHasher()
        >>> hasher
        StringHasher()
        >>> hasher.hash("Meowwwwww", registry=registry)
        '1b06bfa9e842b52eaf47386798687ccd22697ed0198cfda4e0eee7e4650595f5'

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(self, data: str, registry: HasherRegistry, length: int = 64) -> str:  # noqa: ARG002
        return hash_string(data, length=length)


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
    return hashlib.blake2b(value.encode(), digest_size=digest_size).hexdigest()

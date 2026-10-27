r"""Provide deterministic hashing functions for bytes."""

from __future__ import annotations

__all__ = ["BytesHasher", "hash_bytes"]

import hashlib

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, digest, tagged
from coola.hashing.string import validate_hash_length


class BytesHasher(InlineDisplayMixin, BaseHasher[bytes | bytearray | memoryview]):
    r"""Hasher for ``bytes``, ``bytearray`` and ``memoryview`` objects.

    This hasher computes the hash of the bytes namespaced by the type
    tag (``bytes``, ``bytearray`` or ``memoryview``), so a ``bytes``, a
    ``bytearray`` and a ``memoryview`` with the same content hash
    differently.

    Example:
        ```pycon
        >>> from coola.hashing import BytesHasher
        >>> hasher = BytesHasher()
        >>> hasher
        BytesHasher()
        >>> len(hasher.encode(b"hello"))
        70

        ```
    """

    def encode(self, data: bytes | bytearray | memoryview) -> Encoding:
        r"""Describe a ``bytes``, a ``bytearray`` or a ``memoryview``.

        The content is reduced with ``digest`` so that the encoding is
        small, whatever the size of the buffer.

        Args:
            data: The bytes to describe.

        Returns:
            The leaf ``tagged(tag, digest(content))``.
        """
        if isinstance(data, bytearray):
            tag = "bytearray"
        elif isinstance(data, memoryview):
            tag = "memoryview"
        else:
            tag = "bytes"
        return tagged(tag, digest(bytes(data)))


def hash_bytes(value: bytes, length: int = 64) -> str:
    """Generate a hexadecimal hash of bytes using BLAKE2b.

    Args:
        value: The input bytes to hash.
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
        >>> from coola.hashing import hash_bytes
        >>> len(hash_bytes(b"hello", length=16))
        16

        ```
    """
    validate_hash_length(length)
    digest_size = length // 2
    return hashlib.blake2b(value, digest_size=digest_size).hexdigest()

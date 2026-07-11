r"""Provide deterministic hashing functions for bytes."""

from __future__ import annotations

__all__ = ["BytesHasher", "hash_bytes"]

import hashlib
from typing import TYPE_CHECKING

from coola.hashing.base import BaseHasher

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class BytesHasher(BaseHasher[bytes]):
    r"""Hasher for bytes objects.

    This hasher computes the hash of the bytes directly, without any
    intermediate conversion.

    Example:
        ```pycon
        >>> from coola.hashing import BytesHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = BytesHasher()
        >>> hasher
        BytesHasher()
        >>> len(hasher.hash(b"hello", registry=registry))
        64
        >>> len(hasher.hash(b"hello", registry=registry, length=16))
        16

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(self, data: bytes, registry: HasherRegistry, length: int = 64) -> str:  # noqa: ARG002
        r"""Compute a deterministic hash of bytes.

        Args:
            data: The input bytes to hash.
            registry: The hasher registry. Unused by this hasher since
                bytes are hashed directly with no need to dispatch to
                another hasher for nested/composite data; accepted
                only to satisfy the common ``BaseHasher`` interface.
            length: The desired length of the returned hex string. See
                ``hash_bytes`` for constraints. Defaults to 64.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        return hash_bytes(data, length=length)


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
    if length % 2 != 0 or not (2 <= length <= 128):
        msg = f"length must be an even number between 2 and 128, got {length}."
        raise ValueError(msg)
    digest_size = length // 2
    return hashlib.blake2b(value, digest_size=digest_size).hexdigest()

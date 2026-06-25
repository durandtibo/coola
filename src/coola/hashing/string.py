r"""Provide deterministic hashing functions for strings."""

from __future__ import annotations

__all__ = ["hash_string"]

import hashlib


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

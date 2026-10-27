r"""Provide deterministic hashing functions for bytes."""

from __future__ import annotations

__all__ = ["BytesHasher"]


from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, digest, tagged


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

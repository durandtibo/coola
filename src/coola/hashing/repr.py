r"""Define the repr hasher."""

from __future__ import annotations

__all__ = ["ReprHasher"]

from typing import TYPE_CHECKING, Any

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class ReprHasher(BaseHasher[Any]):
    r"""Hasher for objects whose ``repr()`` is a reliable canonical
    representation.

    This hasher converts the object to its ``repr()`` string and then
    computes the hash of that string. It is preferable to
    ``DefaultHasher`` for numeric types (``int``, ``float``, ``complex``,
    ``bool``) because ``repr()`` guarantees round-trip accuracy for
    floating point values, whereas ``str()`` may lose precision on some
    platforms.

    Example:
        ```pycon
        >>> from coola.hashing import ReprHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = ReprHasher()
        >>> hasher
        ReprHasher()
        >>> hasher.hash(1234, registry=registry)
        'bf1003cd5c1336387f7e4eebf72a3d9cd4fa8ab5be19825bc0e3ecd8ce1cd140'

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(self, data: Any, registry: HasherRegistry, length: int = 64) -> str:  # noqa: ARG002
        return hash_string(repr(data), length=length)

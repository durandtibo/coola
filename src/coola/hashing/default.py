r"""Define the default hasher."""

from __future__ import annotations

__all__ = ["DefaultHasher"]

from typing import TYPE_CHECKING, Any

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class DefaultHasher(BaseHasher[Any]):
    r"""Hasher for objects that do not have a specific hasher registered.

    This hasher converts the object to its string representation and
    then computes the hash of that string.

    Example:
        ```pycon
        >>> from coola.hashing import DefaultHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = DefaultHasher()
        >>> hasher
        DefaultHasher()
        >>> hasher.hash("Meowwwwww", registry=registry)

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(self, data: Any, registry: HasherRegistry, length: int = 64) -> str:  # noqa: ARG002
        return hash_string(str(data), length=length)

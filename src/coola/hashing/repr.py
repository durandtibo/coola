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
    ``StrHasher`` for numeric types (``int``, ``float``, ``complex``,
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

    def hash(
        self,
        data: Any,
        registry: HasherRegistry,  # noqa: ARG002
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        r"""Compute a deterministic hash of ``repr(data)``.

        Args:
            data: The object to hash.
            registry: The hasher registry. Unused by this hasher since
                the ``repr()`` string is hashed directly with no need
                to dispatch to another hasher for nested data; accepted
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
        return hash_string(repr(data), length=length)

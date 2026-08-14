r"""Define the str hasher."""

from __future__ import annotations

__all__ = ["StrHasher"]

from typing import TYPE_CHECKING, Any

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class StrHasher(BaseHasher[Any]):
    r"""Hasher for objects whose ``str()`` is a reliable canonical
    representation.

    This hasher converts the object to its ``str()`` string and then
    computes the hash of that string. Unlike ``ReprHasher``, which uses
    ``repr()``, this hasher uses ``str()``, which is more human-readable
    but does not guarantee round-trip accuracy for floating point values.

    Example:
        ```pycon
        >>> from coola.hashing import StrHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = StrHasher()
        >>> hasher
        StrHasher()
        >>> hasher.hash("hello", registry=registry)
        '324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf'

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
        r"""Compute a deterministic hash of ``str(data)``.

        Args:
            data: The object to hash.
            registry: The hasher registry. Unused by this hasher since
                the ``str()`` string is hashed directly with no need to
                dispatch to another hasher for nested data; accepted
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
        return hash_string(str(data), length=length)

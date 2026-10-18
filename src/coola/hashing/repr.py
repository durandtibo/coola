r"""Define the repr hasher."""

from __future__ import annotations

__all__ = ["ReprHasher"]

from enum import Enum
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_enum, hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class ReprHasher(InlineDisplayMixin, NoArgsDisplayMixin, BaseHasher[Any]):
    r"""Hasher for objects whose ``repr()`` is a reliable canonical
    representation.

    This hasher converts the object to its ``repr()`` string and hashes
    it under a type tag. Instances of ``bool``, ``int``, ``float`` and
    ``complex`` (including subclasses such as NumPy scalars) use the
    builtin ``repr`` and the builtin type name as tag, so they hash like
    the equivalent builtin value. Enum members are hashed with
    ``hash_enum`` instead. It is preferable to
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
        >>> len(hasher.hash(1234, registry=registry))
        64

        ```
    """

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
        if isinstance(data, Enum):
            # ``(int, Enum)`` mixins reach this hasher through the MRO.
            return hash_enum(data, length=length)
        for base in (bool, int, float, complex):
            if isinstance(data, base):
                # Use the builtin repr so subclasses (e.g. NumPy scalars)
                # hash like the builtin value, independent of library versions.
                return hash_tagged(base.__name__, base.__repr__(data), length=length)
        return hash_tagged(type(data).__qualname__, repr(data), length=length)

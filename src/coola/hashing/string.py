r"""Provide deterministic hashing functions for strings."""

from __future__ import annotations

__all__ = ["StringHasher"]

from collections import UserString

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, tagged


class StringHasher(InlineDisplayMixin, BaseHasher[str | UserString]):
    r"""Hasher for string objects.

    This hasher describes the string under the ``str`` type tag, without
    any intermediate conversion. ``str`` subclasses hash like plain
    strings (the enum members are dispatched to ``EnumHasher`` by the
    registry). A ``UserString`` is described by its text under the
    ``UserString`` tag, so it differs from a ``str``.

    Example:
        ```pycon
        >>> from coola.hashing import StringHasher
        >>> hasher = StringHasher()
        >>> hasher
        StringHasher()
        >>> hasher.encode("Meowwwwww")
        b'str\x00Meowwwwww'

        ```
    """

    def encode(self, data: str | UserString) -> Encoding:
        r"""Describe a string.

        A lone surrogate (for example in the result of ``os.fsdecode``
        for a file name that is not valid UTF-8) is encoded with the
        ``surrogatepass`` error handler.

        Args:
            data: The string to describe.

        Returns:
            The leaf ``tagged("str", data)``, or
            ``tagged("UserString", str(data))`` for a ``UserString``.
        """
        if isinstance(data, UserString):
            return tagged("UserString", str(data))
        return tagged("str", data)

r"""Define the str hasher."""

from __future__ import annotations

__all__ = ["StrHasher"]

from typing import Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, tagged
from coola.utils.introspection import get_qualified_class_name


class StrHasher(InlineDisplayMixin, BaseHasher[Any]):
    r"""Hasher for objects whose ``str()`` is a reliable canonical
    representation.

    This hasher converts the object to its ``str()`` string and then
    computes the hash of that string. Unlike ``ReprHasher``, which uses
    ``repr()``, this hasher uses ``str()``, which is more human-readable
    but does not guarantee round-trip accuracy for floating point values.

    The tag identifies the type family. By default it is ``str:`` followed
    by the module and the qualified name of the class, so a user class
    never shares the tag of another class with the same name. Pass
    ``tag`` to register a constant tag for a type (``StrHasher("UUID")``):
    a subclass then hashes like its base.

    Args:
        tag: The type tag mixed into the hash. If ``None``, the tag is
            derived from the class of the hashed object. Defaults to
            ``None``.

    Example:
        ```pycon
        >>> from coola.hashing import StrHasher
        >>> hasher = StrHasher()
        >>> hasher
        StrHasher()
        >>> hasher.encode("hello")
        b'str:str\x00hello'

        ```
    """

    def __init__(self, tag: str | None = None) -> None:
        self._tag = tag

    def _display_fields(self) -> dict[str, Any]:
        return {} if self._tag is None else {"tag": self._tag}

    def encode(self, data: Any) -> Encoding:
        r"""Describe an object by its ``str()``.

        Args:
            data: The object to describe.

        Returns:
            The leaf ``tagged(tag, str(data))``.
        """
        tag = self._tag or f"str:{get_qualified_class_name(type(data))}"
        return tagged(tag, str(data))

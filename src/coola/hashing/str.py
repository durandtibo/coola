r"""Define the str hasher."""

from __future__ import annotations

__all__ = ["StrHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged
from coola.utils.introspection import get_qualified_class_name

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


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
        >>> from coola.hashing import StrHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = StrHasher()
        >>> hasher
        StrHasher()
        >>> len(hasher.hash("hello", registry=registry))
        64

        ```
    """

    def __init__(self, tag: str | None = None) -> None:
        self._tag = tag

    def _display_fields(self) -> dict[str, Any]:
        return {} if self._tag is None else {"tag": self._tag}

    def hash(
        self,
        data: Any,
        registry: HasherRegistry,  # noqa: ARG002
        *,
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
        tag = self._tag or f"str:{get_qualified_class_name(type(data))}"
        return hash_tagged(tag, str(data), length=length)

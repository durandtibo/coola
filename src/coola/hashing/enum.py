r"""Define the enum hasher."""

from __future__ import annotations

__all__ = ["EnumHasher"]

from enum import Enum

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, tagged


class EnumHasher(InlineDisplayMixin, BaseHasher[Enum]):
    r"""Hasher for ``enum.Enum`` members.

    The hash depends on the qualified class name and the member name,
    not on the member value. ``IntEnum`` and ``StrEnum`` members are
    therefore not collapsed into ``int`` and ``str``.

    Example:
        ```pycon
        >>> from enum import Enum
        >>> from coola.hashing import EnumHasher
        >>> class Color(Enum):
        ...     RED = 1
        ...
        >>> hasher = EnumHasher()
        >>> hasher
        EnumHasher()
        >>> hasher.encode(Color.RED).startswith(b"Enum")
        True

        ```
    """

    def encode(self, data: Enum) -> Encoding:
        r"""Describe an enum member.

        The encoding depends on the qualified name of the enum class and
        on the member name, not on the member value, so it does not
        collapse ``IntEnum``/``StrEnum`` members into ``int``/``str``.
        A combined flag has no name: its value is used instead.

        Args:
            data: The enum member to describe.

        Returns:
            The leaf ``tagged("Enum", "module.Class.NAME")``.
        """
        cls = type(data)
        name = data.name if data.name is not None else repr(data.value)
        return tagged("Enum", f"{cls.__module__}.{cls.__qualname__}.{name}")

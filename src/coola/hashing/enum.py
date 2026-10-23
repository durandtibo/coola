r"""Define the enum hasher."""

from __future__ import annotations

__all__ = ["EnumHasher"]

from enum import Enum
from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_enum

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class EnumHasher(InlineDisplayMixin, NoArgsDisplayMixin, BaseHasher[Enum]):
    r"""Hasher for ``enum.Enum`` members.

    The hash depends on the qualified class name and the member name,
    not on the member value. ``IntEnum`` and ``StrEnum`` members are
    therefore not collapsed into ``int`` and ``str``.

    Example:
        ```pycon
        >>> from enum import Enum
        >>> from coola.hashing import EnumHasher, HasherRegistry
        >>> class Color(Enum):
        ...     RED = 1
        ...
        >>> registry = HasherRegistry()
        >>> hasher = EnumHasher()
        >>> hasher
        EnumHasher()
        >>> len(hasher.hash(Color.RED, registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: Enum,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        r"""Compute a deterministic hash of an enum member.

        Args:
            data: The enum member to hash.
            registry: The hasher registry. Unused; accepted only to
                satisfy the common ``BaseHasher`` interface.
            length: The desired length of the returned hex string. See
                ``hash_string`` for constraints. Defaults to 64.
            ignore_unhashable: Unused; accepted only to satisfy the
                common ``BaseHasher`` interface.

        Returns:
            A lowercase hexadecimal string of exactly ``length``
            characters.

        Raises:
            ValueError: If ``length`` is not an even number between 2
                and 128.
        """
        return hash_enum(data, length=length)

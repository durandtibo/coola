r"""Define the datetime hasher."""

from __future__ import annotations

__all__ = ["DatetimeHasher"]

from datetime import date, time
from typing import TYPE_CHECKING

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class DatetimeHasher(InlineDisplayMixin, NoArgsDisplayMixin, BaseHasher[date | time]):
    r"""Hasher for :class:`datetime.date`, :class:`datetime.datetime`
    and :class:`datetime.time` objects.

    This hasher converts the object to its ISO 8601 string representation
    via ``isoformat()`` and hashes that string under a tag made of the
    concrete class name.

    Since :class:`datetime.datetime` is a subclass of :class:`datetime.date`,
    both types are handled by this hasher. The class name is part of the
    tag, so a ``date`` and a ``datetime`` never produce the same hash,
    even for the same calendar day.

    Offsets are part of the ISO string, so naive and aware values
    differ, and equal instants in different time zones hash
    differently.

    Example:
        ```pycon
        >>> from datetime import date
        >>> from coola.hashing import DatetimeHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = DatetimeHasher()
        >>> hasher
        DatetimeHasher()
        >>> len(hasher.hash(date(2021, 1, 1), registry=registry))
        64

        ```
    """

    def hash(
        self,
        data: date | time,
        registry: HasherRegistry,  # noqa: ARG002
        *,
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        r"""Compute a deterministic hash of a date/datetime.

        Args:
            data: The date or datetime to hash.
            registry: The hasher registry. Unused by this hasher since
                the ISO string is hashed directly with no need to
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
        return hash_tagged(type(data).__qualname__, data.isoformat(), length=length)

r"""Define the datetime hasher."""

from __future__ import annotations

__all__ = ["DatetimeHasher"]

from datetime import date
from typing import TYPE_CHECKING

from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_string

if TYPE_CHECKING:
    from coola.hashing.registry import HasherRegistry


class DatetimeHasher(BaseHasher[date]):
    r"""Hasher for :class:`datetime.date` and :class:`datetime.datetime`
    objects.

    This hasher converts the object to its ISO 8601 string representation
    via ``isoformat()`` and then computes the hash of that string.

    Since :class:`datetime.datetime` is a subclass of :class:`datetime.date`,
    both types are handled by this hasher. Their ISO representations are
    always distinct (e.g. ``'2021-01-01'`` vs ``'2021-01-01T00:00:00'``),
    so they never produce the same hash for the same calendar date.

    Example:
        ```pycon
        >>> from datetime import date
        >>> from coola.hashing import DatetimeHasher, HasherRegistry
        >>> registry = HasherRegistry()
        >>> hasher = DatetimeHasher()
        >>> hasher
        DatetimeHasher()
        >>> hasher.hash(date(2021, 1, 1), registry=registry)
        'f2b4c6a9941206bb6fc3b4b9c1104d8c05264985c009e2e1c7c840aaeda00dac'

        ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def hash(
        self,
        data: date,
        registry: HasherRegistry,  # noqa: ARG002
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
        return hash_string(data.isoformat(), length=length)

r"""Contain functions to select items of a collection."""

from __future__ import annotations

__all__ = ["smallest_items"]

import heapq
from itertools import islice
from typing import TYPE_CHECKING, Any

from coola.validation import validate_ge

if TYPE_CHECKING:
    from collections.abc import Collection


def smallest_items(data: Collection[Any], *, limit: int | None = None) -> list[Any]:
    r"""Return the smallest items in order.

    The items are selected in one pass without sorting all of them. If
    the items cannot be compared (for example an ``int`` and a ``str``),
    the first items in iteration order are returned instead.

    Args:
        data: The items, for example a set. It is a collection (not an
            iterator) because it is iterated twice when the items cannot
            be compared.
        limit: The maximum number of items to return. ``None`` returns
            all the items.

    Returns:
        The ``limit`` smallest items in order, or the first ``limit``
            items in iteration order if they cannot be compared.

    Raises:
        ValueError: If ``limit`` is negative.

    Example:
        ```pycon
        >>> from coola.utils.selection import smallest_items
        >>> smallest_items({3, 1, 2})
        [1, 2, 3]
        >>> smallest_items({3, 1, 2}, limit=2)
        [1, 2]
        >>> smallest_items([1, "a"], limit=1)
        [1]

        ```
    """
    if limit is not None:
        validate_ge(limit, 0, name="limit")
    try:
        return sorted(data) if limit is None else heapq.nsmallest(limit, data)
    except TypeError:
        return list(islice(data, limit))

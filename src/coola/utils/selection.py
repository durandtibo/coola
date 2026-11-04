r"""Contain functions to select items of a collection."""

from __future__ import annotations

__all__ = ["smallest_items", "sort_items"]

import heapq
from typing import TYPE_CHECKING, Any

from coola.validation import validate_ge

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable


def smallest_items(data: Collection[Any], *, limit: int | None = None) -> list[Any]:
    r"""Return the smallest items in order.

    The items are selected in one pass without sorting all of them. If
    the items cannot be compared (for example an ``int`` and a ``str``),
    they are sorted with the fallbacks of
    [`sort_items`][coola.utils.selection.sort_items], so the result does
    not depend on the iteration order of ``data``.

    Args:
        data: The items, for example a set. It is a collection (not an
            iterator) because it is iterated twice when the items cannot
            be compared.
        limit: The maximum number of items to return. ``None`` returns
            all the items.

    Returns:
        The ``limit`` smallest items in order.

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
        >>> smallest_items({1, "a"})
        [1, 'a']

        ```
    """
    if limit is not None:
        validate_ge(limit, 0, name="limit")
    try:
        return sorted(data) if limit is None else heapq.nsmallest(limit, data)
    except TypeError:
        return sort_items(data)[:limit]


def sort_items(data: Iterable[Any], *, key: Callable[[Any], Any] | None = None) -> list[Any]:
    r"""Sort items that may not be comparable with each other.

    The items are sorted by ``key(item)`` (or by the item if ``key`` is
    ``None``). If the sorted values cannot be compared with each other
    (for example ``1`` and ``"a"``), they are sorted by type name first,
    then by value. If two values of the same type cannot be compared
    either (for example the tuples ``(1,)`` and ``("a",)``), they are
    sorted by type name first, then by ``repr`` of the value. The result
    does not depend on the order of ``data``, so it can be used for a
    set.

    Args:
        data: The items to sort.
        key: A function that returns the value to sort by, or ``None``
            to sort the items themselves.

    Returns:
        The sorted items.

    Example:
        ```pycon
        >>> from coola.utils.selection import sort_items
        >>> sort_items({3, 1, 2})
        [1, 2, 3]
        >>> sort_items([2, "b", 1, "a"])
        [1, 2, 'a', 'b']
        >>> sort_items([("b", 1), ("a", 2)], key=lambda x: x[0])
        [('a', 2), ('b', 1)]

        ```
    """
    items = list(data)
    value = key if key is not None else (lambda item: item)
    try:
        return sorted(items, key=value)
    except TypeError:
        pass
    try:
        return sorted(items, key=lambda item: (type(value(item)).__name__, value(item)))
    except TypeError:
        return sorted(items, key=lambda item: (type(value(item)).__name__, repr(value(item))))

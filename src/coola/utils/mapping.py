r"""Contain utility functions for mappings."""

from __future__ import annotations

__all__ = ["sort_by_keys", "sort_by_values"]

import operator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def sort_by_keys(mapping: Mapping[Any, Any]) -> dict[Any, Any]:
    r"""Sort a dictionary by keys.

    If the keys cannot be compared with each other (for example ``1``
    and ``"a"``), they are sorted by type name first, then by key. If
    two keys of the same type cannot be compared either (for example
    the tuples ``(1,)`` and ``("a",)``), they are sorted by type name
    first, then by ``repr`` of the key.

    Args:
        mapping: The dictionary to sort.

    Returns:
        The sorted dictionary.

    Example:
        ```pycon
        >>> from coola.utils.mapping import sort_by_keys
        >>> sort_by_keys({"dog": 1, "cat": 5, "fish": 2})
        {'cat': 5, 'dog': 1, 'fish': 2}

        ```
    """
    try:
        return dict(sorted(mapping.items(), key=operator.itemgetter(0)))
    except TypeError:
        pass
    try:
        return dict(sorted(mapping.items(), key=lambda item: (type(item[0]).__name__, item[0])))
    except TypeError:
        return dict(
            sorted(mapping.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        )


def sort_by_values(mapping: Mapping[Any, Any]) -> dict[Any, Any]:
    r"""Sort a dictionary by values.

    Args:
        mapping: The dictionary to sort.

    Returns:
        The sorted dictionary.

    Example:
        ```pycon
        >>> from coola.utils.mapping import sort_by_values
        >>> sort_by_values({"dog": 1, "cat": 5, "fish": 2})
        {'dog': 1, 'fish': 2, 'cat': 5}

        ```
    """
    return dict(sorted(mapping.items(), key=operator.itemgetter(1)))

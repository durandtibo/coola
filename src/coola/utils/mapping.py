r"""Contain utility functions for mappings."""

from __future__ import annotations

__all__ = ["sort_by_keys", "sort_by_values"]

import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def sort_by_keys(mapping: Mapping) -> dict:
    r"""Sort a dictionary by keys.

    Args:
        mapping: The dictionary to sort.

    Returns:
        dict: The sorted dictionary.

    Example usage:

    ```pycon

    >>> from coola.utils.mapping import sort_by_keys
    >>> sort_by_keys({"dog": 1, "cat": 5, "fish": 2})
    {'cat': 5, 'dog': 1, 'fish': 2}

    ```
    """
    return dict(sorted(mapping.items()))


def sort_by_values(mapping: Mapping) -> dict:
    r"""Sort a dictionary by keys.

    Args:
        mapping: The dictionary to sort.

    Returns:
        dict: The sorted dictionary.

    Example usage:

    ```pycon

    >>> from coola.utils.mapping import sort_by_values
    >>> sort_by_values({"dog": 1, "cat": 5, "fish": 2})
    {'dog': 1, 'fish': 2, 'cat': 5}

    ```
    """
    return dict(sorted(mapping.items(), key=operator.itemgetter(1)))

r"""Contain some utility functions to manipulate mappings."""

from __future__ import annotations

__all__ = ["get_first_value", "remove_keys_starting_with"]

from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Mapping

T = TypeVar("T")


def get_first_value(data: Mapping[Any, T]) -> T:
    r"""Get the first value of a mapping.

    Args:
        data: The input mapping.

    Returns:
        The first value in the mapping.

    Raises:
        ValueError: if the mapping is empty.

    Example:
        ```pycon
        >>> from coola.nested import get_first_value
        >>> get_first_value({"key1": 1, "key2": 2})
        1

        ```
    """
    if not data:
        msg = "First value cannot be returned because the mapping is empty"
        raise ValueError(msg)
    return data[next(iter(data))]


def remove_keys_starting_with(mapping: Mapping[Any, Any], prefix: str) -> dict[Any, Any]:
    r"""Remove the keys that start with a given prefix.

    Args:
        mapping: The original mapping.
        prefix: The prefix used to filter the keys.

    Returns:
        A new dict without the removed keys.

    Example:
        ```pycon
        >>> from coola.nested import remove_keys_starting_with
        >>> remove_keys_starting_with(
        ...     {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6},
        ...     "key",
        ... )
        {'abc': 3, 'abc.key': 4, 1: 5, (2, 3): 6}

        ```
    """
    out = {}
    for key, value in mapping.items():
        if isinstance(key, str) and key.startswith(prefix):
            continue
        out[key] = value
    return out

r"""Contain some utility functions to manipulate mappings."""

from __future__ import annotations

__all__ = ["get_first_value", "merge_list_of_mappings", "remove_keys_starting_with"]

from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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


def merge_list_of_mappings(
    mappings: Iterable[Mapping[Any, Any]], on_duplicate: str = "raise"
) -> dict[Any, Any]:
    r"""Merge an iterable of mappings into a single dict.

    Args:
        mappings: The mappings to merge.
        on_duplicate: The strategy used to manage duplicate keys.
            The valid values are:
            - ``'raise'``: raise an exception if a key appears
                more than once.
            - ``'first'``: keep the value from the first occurrence.
            - ``'last'``: keep the value from the last occurrence.
            - ``'suffix'``: keep all the values by adding a
                ``'_n'`` suffix to the key name, where ``n`` is
                incremented for each new occurrence.

    Returns:
        The merged dict.

    Raises:
        ValueError: if ``on_duplicate`` is not a valid value.
        KeyError: if ``on_duplicate='raise'`` and a key appears
            more than once.

    Example:
        ```pycon
        >>> from coola.nested import merge_list_of_mappings
        >>> merge_list_of_mappings(
        ...     [{"key1": 1, "key2": 2}, {"key2": 3, "key3": 4}], on_duplicate="last"
        ... )
        {'key1': 1, 'key2': 3, 'key3': 4}

        ```
    """
    strategies = {"raise", "first", "last", "suffix"}
    if on_duplicate not in strategies:
        msg = f"Incorrect on_duplicate value: {on_duplicate!r}. Valid values are {strategies}"
        raise ValueError(msg)

    out: dict[Any, Any] = {}
    counts: dict[Any, int] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key not in counts:
                counts[key] = 1
                out[key] = value
                continue

            counts[key] += 1
            if on_duplicate == "raise":
                msg = f"Duplicate key found: {key!r}"
                raise KeyError(msg)
            if on_duplicate == "last":
                out[key] = value
            elif on_duplicate == "suffix":
                out[f"{key}_{counts[key] - 1}"] = value
            # on_duplicate == "first" -> keep existing value, do nothing

    return out


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

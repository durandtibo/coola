r"""Contain some utility functions to manipulate mappings."""

from __future__ import annotations

__all__ = [
    "flatten_mapping",
    "get_first_value",
    "merge_mappings",
    "remove_keys_starting_with",
]

from typing import TYPE_CHECKING, Any, TypeVar

from coola.equality import objects_are_equal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

T = TypeVar("T")


def flatten_mapping(
    mapping: Mapping[Any, Mapping[Any, Any]],
    on_duplicate: str = "raise",
    always_prefix: bool = False,
    separator: str = ".",
) -> dict[Any, Any]:
    r"""Flatten a mapping of mappings into a single dict.

    Each outer key is used as a prefix for its inner keys. If an
    inner key appears in more than one outer mapping with the same
    value in every occurrence, the plain inner key is kept and
    ``on_duplicate`` is not triggered. ``on_duplicate`` only applies
    when an inner key appears more than once with different values.

    Args:
        mapping: The mapping of mappings to flatten.
        on_duplicate: The strategy used to manage duplicate inner
            keys that have different values across outer mappings.
            The valid values are:
            - ``'raise'``: raise an exception if an inner key
                appears more than once with different values.
            - ``'first'``: keep the value from the first occurrence,
                under the plain inner key.
            - ``'last'``: keep the value from the last occurrence,
                under the plain inner key.
            - ``'prefix'``: keep all the values, renaming every
                occurrence of a conflicting key as
                ``'{outer_key}{separator}{inner_key}'``.
        always_prefix: If ``True``, every key in the output is named
            ``'{outer_key}{separator}{inner_key}'``, regardless of
            whether it is a duplicate. If ``False``, only keys
            involved in a genuine conflict are prefixed (only
            relevant when ``on_duplicate='prefix'``).
        separator: The separator used to join the outer and inner
            keys when prefixing.

    Returns:
        The flattened dict.

    Raises:
        ValueError: if ``on_duplicate`` is not a valid value.
        KeyError: if ``on_duplicate='raise'`` and an inner key
            appears more than once with different values.

    Example:
        ```pycon
        >>> from coola.nested import flatten_mapping
        >>> flatten_mapping(
        ...     {"module1": {"a": 1, "b": 2}, "module2": {"b": 3, "c": 4}},
        ...     on_duplicate="prefix",
        ... )
        {'a': 1, 'module1.b': 2, 'module2.b': 3, 'c': 4}

        ```
    """
    strategies = {"raise", "first", "last", "prefix"}
    if on_duplicate not in strategies:
        msg = f"Incorrect on_duplicate value: {on_duplicate!r}. Valid values are {strategies}"
        raise ValueError(msg)

    out: dict[Any, Any] = {}
    origins: dict[Any, Any] = {}  # inner key -> outer key of the recorded occurrence
    seen: dict[Any, Any] = {}  # inner key -> recorded value, to check for real conflicts

    for outer_key, inner_mapping in mapping.items():
        for inner_key, value in inner_mapping.items():
            prefixed_key = f"{outer_key}{separator}{inner_key}"

            if always_prefix:
                out[prefixed_key] = value
                continue

            if inner_key not in seen:
                seen[inner_key] = value
                origins[inner_key] = outer_key
                out[inner_key] = value
                continue

            if objects_are_equal(seen[inner_key], value):
                continue

            if on_duplicate == "raise":
                msg = f"Duplicate key found: {inner_key!r}"
                raise KeyError(msg)
            if on_duplicate == "first":
                continue
            if on_duplicate == "last":
                seen[inner_key] = value
                out[inner_key] = value
                continue

            # Re-key the first occurrence too, the first time a real
            # conflict is found for this inner key.
            if inner_key in out:
                first_key = f"{origins[inner_key]}{separator}{inner_key}"
                out[first_key] = out.pop(inner_key)
            out[prefixed_key] = value

    return out


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


def merge_mappings(
    mappings: Iterable[Mapping[Any, Any]], on_duplicate: str = "raise"
) -> dict[Any, Any]:
    r"""Merge an iterable of mappings into a single dict.

    If a key appears more than once with the same value in every
    occurrence, the value is kept as-is and ``on_duplicate`` is not
    triggered. ``on_duplicate`` only applies when a key appears more
    than once with different values.

    Args:
        mappings: The mappings to merge.
        on_duplicate: The strategy used to manage duplicate keys that
            have different values across mappings.
            The valid values are:
            - ``'raise'``: raise an exception if a key appears
                more than once with different values.
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
            more than once with different values.

    Example:
        ```pycon
        >>> from coola.nested import merge_mappings
        >>> merge_mappings([{"key1": 1, "key2": 2}, {"key2": 3, "key3": 4}], on_duplicate="last")
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

            if objects_are_equal(out[key], value):
                # Same value repeated: not a real conflict, keep as-is.
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

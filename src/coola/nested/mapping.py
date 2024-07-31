r"""Contain some utility functions to manipulate mappings."""

from __future__ import annotations

__all__ = ["get_first_value", "remove_keys_starting_with", "to_flat_dict"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def get_first_value(data: Mapping) -> Any:
    r"""Get the first value of a mapping.

    Args:
        data: The input mapping.

    Returns:
        The first value in the mapping.

    Raises:
        ValueError: if the mapping is empty.

    Example usage:

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


def to_flat_dict(
    data: Any,
    prefix: str | None = None,
    separator: str = ".",
    to_str: type[object] | tuple[type[object], ...] | None = None,
) -> dict[str, Any]:
    r"""Return a flat representation of a nested dict with the dot
    format.

    Args:
        data: The nested dict to flat.
        prefix: The prefix to use to generate the name of the key.
            ``None`` means no prefix.
        separator: The separator to concatenate keys of nested
            collections.
        to_str: The data types which will not be flattened out,
            instead converted to string.

    Returns:
        The flatted dictionary.

    Example usage:

    ```pycon

    >>> from coola.nested import to_flat_dict
    >>> data = {
    ...     "str": "def",
    ...     "module": {
    ...         "component": {
    ...             "float": 3.5,
    ...             "int": 2,
    ...         },
    ...     },
    ... }
    >>> to_flat_dict(data)
    {'str': 'def', 'module.component.float': 3.5, 'module.component.int': 2}
    >>> # Example with lists (also works with tuple)
    >>> data = {
    ...     "module": [[1, 2, 3], {"bool": True}],
    ...     "str": "abc",
    ... }
    >>> to_flat_dict(data)
    {'module.0.0': 1, 'module.0.1': 2, 'module.0.2': 3, 'module.1.bool': True, 'str': 'abc'}
    >>> # Example with lists with to_str=(list) (also works with tuple)
    >>> data = {
    ...     "module": [[1, 2, 3], {"bool": True}],
    ...     "str": "abc",
    ... }
    >>> to_flat_dict(data)
    {'module.0.0': 1, 'module.0.1': 2, 'module.0.2': 3, 'module.1.bool': True, 'str': 'abc'}

    ```
    """
    flat_dict = {}
    to_str = to_str or ()
    if isinstance(data, to_str):
        flat_dict[prefix] = str(data)
    elif isinstance(data, dict):
        for key, value in data.items():
            flat_dict.update(
                to_flat_dict(
                    value,
                    prefix=f"{prefix}{separator}{key}" if prefix else key,
                    separator=separator,
                    to_str=to_str,
                )
            )
    elif isinstance(data, (list, tuple)):
        for i, value in enumerate(data):
            flat_dict.update(
                to_flat_dict(
                    value,
                    prefix=f"{prefix}{separator}{i}" if prefix else str(i),
                    separator=separator,
                    to_str=to_str,
                )
            )
    else:
        flat_dict[prefix] = data
    return flat_dict


def remove_keys_starting_with(mapping: Mapping, prefix: str) -> dict:
    r"""Remove the keys that start with a given prefix.

    Args:
        mapping: The original mapping.
        prefix: The prefix used to filter the keys.

    Returns:
        A new dict without the removed keys.

    Example usage:

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

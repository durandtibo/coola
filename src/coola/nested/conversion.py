r"""Contain some utility functions to convert nested data structure."""

from __future__ import annotations

__all__ = ["convert_to_dict_of_lists", "convert_to_jsonable", "convert_to_list_of_dicts"]

from typing import TYPE_CHECKING, Any

from coola.recursive import recursive_apply
from coola.utils.conversion import to_jsonable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def convert_to_dict_of_lists(
    seq_of_mappings: Sequence[Mapping[Any, Any]],
) -> dict[Any, list[Any]]:
    r"""Convert a sequence of mappings to a dictionary of lists.

    All the mappings must have the same keys as the first mapping in
    the sequence, which is used to find the keys.

    Args:
        seq_of_mappings: The sequence of mappings to convert.

    Returns:
        A dictionary of lists.

    Raises:
        ValueError: If a mapping's keys differ from the first
            mapping's keys.

    Example:
        ```pycon
        >>> from coola.nested import convert_to_dict_of_lists
        >>> convert_to_dict_of_lists(
        ...     [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}]
        ... )
        {'key1': [1, 2, 3], 'key2': [10, 20, 30]}

        ```
    """
    if not seq_of_mappings:
        return {}
    keys = set(seq_of_mappings[0])
    for i, mapping in enumerate(seq_of_mappings):
        if set(mapping) != keys:
            msg = (
                f"All the mappings must have the same keys as the first mapping "
                f"({sorted(keys, key=str)}), but mapping at index {i} has keys "
                f"{sorted(mapping, key=str)}"
            )
            raise ValueError(msg)
    return {key: [dic[key] for dic in seq_of_mappings] for key in seq_of_mappings[0]}


def convert_to_list_of_dicts(
    mapping_of_seqs: Mapping[Any, Sequence[Any]],
) -> list[dict[Any, Any]]:
    r"""Convert a mapping of sequences to a list of dictionaries.

    All the sequences must have the same length.

    Args:
        mapping_of_seqs: The mapping of sequences to convert.

    Returns:
        A list of dictionaries.

    Raises:
        ValueError: If the sequences do not all have the same length.

    Example:
        ```pycon
        >>> from coola.nested import convert_to_list_of_dicts
        >>> convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]})
        [{'key1': 1, 'key2': 10}, {'key1': 2, 'key2': 20}, {'key1': 3, 'key2': 30}]

        ```
    """
    lengths = {key: len(seq) for key, seq in mapping_of_seqs.items()}
    if len(set(lengths.values())) > 1:
        msg = f"All the sequences must have the same length, but received lengths {lengths}"
        raise ValueError(msg)
    return [dict(zip(mapping_of_seqs, seqs)) for seqs in zip(*mapping_of_seqs.values())]


def convert_to_jsonable(data: Any) -> Any:
    r"""Recursively convert a nested data structure to a JSON-compatible
    representation.

    This function walks through nested containers (e.g. ``list``,
    ``tuple``, ``dict``) and applies ``coola.utils.conversion.to_jsonable``
    to every object, converting ``pydantic.BaseModel`` and dataclass
    objects found at any depth. Use ``to_jsonable`` directly if the data
    is a single, non-nested object.

    Args:
        data: The nested data structure to convert.

    Returns:
        The converted data, with the same structure as the input.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.nested import convert_to_jsonable
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> convert_to_jsonable([Point(x=1, y=2), {"key": Point(x=3, y=4)}])
        [{'x': 1, 'y': 2}, {'key': {'x': 3, 'y': 4}}]

        ```
    """
    return recursive_apply(data, to_jsonable)

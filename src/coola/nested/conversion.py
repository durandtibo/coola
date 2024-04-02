r"""Contain some utility functions to convert nested data structure."""

from __future__ import annotations

__all__ = ["convert_to_dict_of_lists", "convert_to_list_of_dicts"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence


def convert_to_dict_of_lists(seq_of_mappings: Sequence[Mapping]) -> dict[Hashable, list]:
    r"""Convert a sequence of mappings to a dictionary of lists.

    All the dictionaries should have the same keys. The first
    mapping in the sequence is used to find the keys.

    Args:
        seq_of_mappings: The sequence of mappings to convert.

    Returns:
        A dictionary of lists.

    Example usage:

    ```pycon

    >>> from coola.nested import convert_to_dict_of_lists
    >>> convert_to_dict_of_lists(
    ...     [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}]
    ... )
    {'key1': [1, 2, 3], 'key2': [10, 20, 30]}

    ```
    """
    if seq_of_mappings:
        return {key: [dic[key] for dic in seq_of_mappings] for key in seq_of_mappings[0]}
    return {}


def convert_to_list_of_dicts(mapping_of_seqs: Mapping[Hashable, Sequence]) -> list[dict]:
    r"""Convert a mapping of sequences to a list of dictionaries.

    All the sequences should have the same length.

    Args:
        mapping_of_seqs: The mapping of sequences to convert.

    Returns:
        A dictionary of lists.

    Example usage:

    ```pycon

    >>> from coola.nested import convert_to_list_of_dicts
    >>> convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]})
    [{'key1': 1, 'key2': 10}, {'key1': 2, 'key2': 20}, {'key1': 3, 'key2': 30}]

    ```
    """
    return [dict(zip(mapping_of_seqs, seqs)) for seqs in zip(*mapping_of_seqs.values())]

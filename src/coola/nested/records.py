r"""Contain some utility functions to convert records and columns."""

from __future__ import annotations

__all__ = ["convert_to_dict_of_lists", "convert_to_list_of_dicts"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def convert_to_dict_of_lists(
    seq_of_mappings: Iterable[Mapping[Any, Any]],
) -> dict[Any, list[Any]]:
    r"""Convert an iterable of mappings to a dictionary of lists.

    All the mappings must have the same keys as the first mapping,
    which is used to find the keys. The order of the keys is the one of
    the first mapping. An iterable that is not a sequence (a generator)
    is consumed once.

    Args:
        seq_of_mappings: The iterable of mappings to convert.

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
    seq_of_mappings = list(seq_of_mappings)
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

    All the sequences must have the same length. A string (or bytes) is
    not accepted as a sequence, because it would be split per character,
    and a set is not accepted either, because its elements have no
    position (the rows of several sets would be paired in an arbitrary
    order). A value must be a ``Sequence`` (a generator is not one).
    An empty mapping gives an empty list, so a list of empty
    dictionaries (``[{}, {}]``) does not round trip with
    ``convert_to_dict_of_lists``.

    Args:
        mapping_of_seqs: The mapping of sequences to convert.

    Returns:
        A list of dictionaries.

    Raises:
        TypeError: If a value is a ``str``, ``bytes`` or ``bytearray``,
            a set, or is not a ``Sequence``.
        ValueError: If the sequences do not all have the same length.

    Example:
        ```pycon
        >>> from coola.nested import convert_to_list_of_dicts
        >>> convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]})
        [{'key1': 1, 'key2': 10}, {'key1': 2, 'key2': 20}, {'key1': 3, 'key2': 30}]

        ```
    """
    for key, seq in mapping_of_seqs.items():
        if isinstance(seq, (str, bytes, bytearray)) or not isinstance(seq, Sequence):
            msg = f"The value of {key!r} must be a sequence of items, not a {type(seq).__name__}"
            raise TypeError(msg)
    lengths = {key: len(seq) for key, seq in mapping_of_seqs.items()}
    if len(set(lengths.values())) > 1:
        msg = f"All the sequences must have the same length, but received lengths {lengths}"
        raise ValueError(msg)
    return [
        dict(zip(mapping_of_seqs, seqs, strict=True))
        for seqs in zip(*mapping_of_seqs.values(), strict=True)
    ]

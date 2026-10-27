r"""Contain some utility functions to convert nested data structure."""

from __future__ import annotations

__all__ = [
    "convert_to_dict_of_lists",
    "convert_to_jsonable",
    "convert_to_list_of_dicts",
    "try_sorted_list",
]

import dataclasses
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.recursive import (
    BaseTransformer,
    DefaultTransformer,
    get_default_registry,
    recursive_apply,
)
from coola.utils.conversion import to_jsonable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence, Set as AbstractSet

    from coola.recursive.base import Expansion


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

    All the sequences must have the same length. A string (or bytes) is
    not accepted as a sequence, because it would be split per character.
    An empty mapping gives an empty list, so a list of empty
    dictionaries (``[{}, {}]``) does not round trip with
    ``convert_to_dict_of_lists``.

    Args:
        mapping_of_seqs: The mapping of sequences to convert.

    Returns:
        A list of dictionaries.

    Raises:
        TypeError: If a value is a ``str``, ``bytes`` or ``bytearray``.
        ValueError: If the sequences do not all have the same length.

    Example:
        ```pycon
        >>> from coola.nested import convert_to_list_of_dicts
        >>> convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]})
        [{'key1': 1, 'key2': 10}, {'key1': 2, 'key2': 20}, {'key1': 3, 'key2': 30}]

        ```
    """
    for key, seq in mapping_of_seqs.items():
        if isinstance(seq, (str, bytes, bytearray)):
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


def convert_to_jsonable(data: Any) -> Any:
    r"""Recursively convert a nested data structure to a JSON-compatible
    representation.

    This function walks through nested containers (e.g. ``list``,
    ``tuple``, ``dict``) and applies ``coola.utils.conversion.to_jsonable``
    to every object, converting ``pydantic.BaseModel`` and dataclass
    objects found at any depth (see ``to_jsonable`` for the supported
    types and the limits: keys are not converted, ``NaN`` is kept).
    Sets are converted to sorted lists. Use ``to_jsonable`` directly if
    the data is a single, non-nested object.

    The contract is "accepted by the Python ``json`` module", not
    "strict JSON": the keys of the dictionaries are not converted
    (``json.dumps`` raises a ``TypeError`` for a key such as a tuple,
    and turns an ``int`` key into a string), and ``NaN`` and infinite
    floats are kept (``json.dumps(..., allow_nan=False)`` raises a
    ``ValueError``). ``coola.utils.conversion.to_jsonable`` converts one
    object and leaves the containers unchanged; this function applies
    it to every object of a nested structure.

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
    registry = get_default_registry().copy()
    # A dataclass is converted whole by ``to_jsonable``, it is not entered.
    registry.register_fallback(dataclasses.is_dataclass, DefaultTransformer(), exist_ok=True)
    transformer = SetToListTransformer()
    registry.register(set, transformer, exist_ok=True)
    registry.register(frozenset, transformer, exist_ok=True)
    return recursive_apply(data, to_jsonable, registry=registry)


class SetToListTransformer(InlineDisplayMixin, BaseTransformer["AbstractSet[Any]"]):
    r"""Transformer for sets that rebuilds a list.

    The elements are the children, so they are converted like the other
    values, and the converted elements are collected in a list sorted
    with ``try_sorted_list``.
    """

    def expand(self, data: AbstractSet[Any]) -> Expansion | None:
        return list(data), try_sorted_list


def try_sorted_list(values: Iterable[Any]) -> list[Any]:
    r"""Convert an iterable (e.g. a set) to a list, sorted when the
    values are comparable so the output is deterministic.

    Args:
        values: The values to convert.

    Returns:
        The values as a sorted list, or in iteration order if they
            cannot be compared.

    Example:
        ```pycon
        >>> from coola.nested import try_sorted_list
        >>> try_sorted_list({3, 1, 2})
        [1, 2, 3]
        >>> try_sorted_list({1, "a"}) == [1, "a"] or try_sorted_list({1, "a"}) == ["a", 1]
        True

        ```
    """
    try:
        return sorted(values)
    except TypeError:
        return list(values)

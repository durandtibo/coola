r"""Contain some utility functions to convert nested data structure."""

from __future__ import annotations

__all__ = [
    "convert_to_dict_of_lists",
    "convert_to_jsonable",
    "convert_to_list_of_dicts",
    "to_jsonable",
]

import base64
import dataclasses
import datetime
import sys
from collections.abc import Sequence
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import TYPE_CHECKING, Any
from uuid import UUID

from coola.display import InlineDisplayMixin
from coola.recursive import (
    BaseTransformer,
    get_default_registry,
    recursive_apply,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Set as AbstractSet

    from coola.recursive.base import Expansion


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


def convert_to_jsonable(data: Any) -> Any:
    r"""Recursively convert a nested data structure to a JSON-compatible
    representation.

    This function walks through nested containers (e.g. ``list``,
    ``tuple``, ``dict``) with the engine of ``coola.recursive`` and
    applies ``to_jsonable`` to every leaf,
    converting ``pydantic.BaseModel`` objects found at any depth (see
    ``to_jsonable`` for the supported types and the limits: keys are
    not converted, ``NaN`` is kept). A dataclass is converted to a
    ``dict`` field by field, like any other container, so its fields
    get the same treatment as everything else. A tuple stays a tuple.
    Sets are converted to sorted lists; the sort does not depend on the
    iteration order of the set, so the output is the same in every
    process. Use ``to_jsonable`` directly if the data is a single,
    non-nested object.

    The contract is "accepted by the Python ``json`` module", not
    "strict JSON": the keys of the dictionaries are not converted
    (``json.dumps`` raises a ``TypeError`` for a key such as a tuple,
    and turns an ``int`` key into a string), and ``NaN`` and infinite
    floats are kept (``json.dumps(..., allow_nan=False)`` raises a
    ``ValueError``). ``to_jsonable`` converts one
    object and leaves the containers unchanged; this function applies
    it to every object of a nested structure.

    Args:
        data: The nested data structure to convert.

    Returns:
        The converted data, with the same structure as the input.

    Raises:
        ValueError: If a container contains itself.

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
    registry.register_fallback(
        dataclasses.is_dataclass, _DataclassToDictTransformer(), exist_ok=True
    )
    transformer = _SetToListTransformer()
    registry.register(set, transformer, exist_ok=True)
    registry.register(frozenset, transformer, exist_ok=True)
    return recursive_apply(data, to_jsonable, registry=registry)


class _DataclassToDictTransformer(InlineDisplayMixin, BaseTransformer[Any]):
    r"""Transformer for dataclass instances that rebuilds a ``dict``.

    The values of the fields are the children, so they are converted
    like the other values, and the converted values are collected in a
    ``dict`` that maps the names of the fields to them.
    """

    def expand(self, data: Any) -> Expansion | None:
        r"""Expand a dataclass instance.

        Args:
            data: The dataclass instance.

        Returns:
            The values of the fields, and the function that builds a
                ``dict`` from the converted values.
        """
        names = [field.name for field in dataclasses.fields(data)]
        return [getattr(data, name) for name in names], lambda items: dict(
            zip(names, items, strict=True)
        )


class _SetToListTransformer(InlineDisplayMixin, BaseTransformer["AbstractSet[Any]"]):
    r"""Transformer for sets that rebuilds a list.

    The elements are the children, so they are converted like the other
    values, and the converted elements are collected in a list sorted
    with ``_sorted_elements``.
    """

    def expand(self, data: AbstractSet[Any]) -> Expansion | None:
        r"""Expand a set.

        Args:
            data: The set.

        Returns:
            The elements, and the function that builds a sorted ``list``
                from the converted elements.
        """
        return list(data), _sorted_elements


def _sort_key(value: Any) -> tuple[str, str]:
    r"""Return the key used to sort values that cannot be compared.

    Args:
        value: The value.

    Returns:
        The ``(type name, repr)`` pair.
    """
    return type(value).__name__, repr(value)


def _sorted_elements(values: Iterable[Any]) -> list[Any]:
    r"""Convert the elements of a set to a sorted list.

    The sort never depends on the iteration order of the set: the
    elements are sorted by value when they can be compared, otherwise
    (elements of different types, ``NaN``) by ``(type name, repr)``.

    Args:
        values: The elements to sort.

    Returns:
        The elements as a sorted list.
    """
    elements = list(values)
    # The comparisons with NaN are always false, so a sort by value depends on the input order.
    if not any(value != value for value in elements):  # noqa: PLR0124
        try:
            return sorted(elements)
        except TypeError:
            pass
    return sorted(elements, key=_sort_key)


def to_jsonable(data: Any) -> Any:
    r"""Convert a single object to a JSON-compatible representation.

    This function converts ``pydantic.BaseModel``,
    ``numpy.ndarray``, ``numpy`` scalar, ``torch.Tensor``,
    ``pathlib`` path, ``datetime.date``/``datetime.datetime``/``datetime.time``,
    ``datetime.timedelta``,
    ``decimal.Decimal``, ``uuid.UUID``, ``bytes``/``bytearray`` and
    ``enum.Enum`` and ``range`` objects. Paths, decimals and UUIDs are
    converted to strings, ranges to lists, dates and times to ISO 8601 strings,
    timedeltas to a number of seconds,
    bytes to a Base64 string
    (ASCII) and enum members to their (converted) value.
    All other object types, including containers such as ``list`` or
    ``dict`` and dataclasses, are returned unchanged: use
    ``convert_to_jsonable`` for a dataclass. Non-string
    ``dict`` keys are not converted, so the result
    is accepted by ``json.dumps`` only if the keys are ``str``, ``int``,
    ``float``, ``bool`` or ``None``; ``NaN`` and infinite floats are
    kept as they are. Sets are not converted (use
    ``convert_to_jsonable``, which converts them to sorted
    lists). To recursively convert every object in a nested data
    structure, use ``convert_to_jsonable`` instead. The
    contract is "accepted by the Python ``json`` module", not "strict
    JSON" (see the notes on the keys and on ``NaN`` above).

    Args:
        data: The object to convert.

    Returns:
        The converted object if it is one of the supported types,
            otherwise the object unchanged.

    Example:
        ```pycon
        >>> from pathlib import PurePosixPath
        >>> from coola.nested.conversion import to_jsonable
        >>> to_jsonable(PurePosixPath("a/b"))
        'a/b'
        >>> to_jsonable([PurePosixPath("a/b")])  # not converted because it is a list
        [PurePosixPath('a/b')]

        ```
    """
    for matches, convert in _CONVERTERS:
        if matches(data):
            return convert(data)
    return data


def _is_instance_of_loaded(data: Any, library: str, *names: str) -> bool:
    r"""Indicate if ``data`` is an instance of one of the types of a
    library.

    An object cannot be an instance of a type of a library that is not
    imported, so the library is not imported here: this check does not
    add the import time of the optional libraries to ``coola``.

    Args:
        data: The object to check.
        library: The name of the library.
        *names: The names of the types of the library.

    Returns:
        ``True`` if the library is imported and ``data`` is an instance
            of one of the types.
    """
    module = sys.modules.get(library)
    return module is not None and isinstance(data, tuple(getattr(module, name) for name in names))


def _is_array(data: Any) -> bool:
    r"""Indicate if ``data`` is a NumPy array or scalar, or a
    ``torch.Tensor``.

    Args:
        data: The object to check.

    Returns:
        ``True`` if ``data`` is an array or a tensor, otherwise
            ``False``.
    """
    return _is_instance_of_loaded(data, "numpy", "ndarray", "generic") or _is_instance_of_loaded(
        data, "torch", "Tensor"
    )


# (predicate, converter) pairs, checked in order by ``to_jsonable``.
_CONVERTERS: tuple[tuple[Callable[[Any], bool], Callable[[Any], Any]], ...] = (
    (
        lambda data: _is_instance_of_loaded(data, "pydantic", "BaseModel"),
        lambda data: data.model_dump(mode="json"),
    ),
    (_is_array, lambda data: data.tolist()),
    (lambda data: isinstance(data, (PurePath, Decimal, UUID)), str),
    (
        lambda data: isinstance(data, (bytes, bytearray)),
        lambda data: base64.b64encode(data).decode("ascii"),
    ),
    (lambda data: isinstance(data, (datetime.date, datetime.time)), lambda data: data.isoformat()),
    (lambda data: isinstance(data, datetime.timedelta), lambda data: data.total_seconds()),
    (lambda data: isinstance(data, Enum), lambda data: to_jsonable(data.value)),
    (lambda data: isinstance(data, range), list),
)

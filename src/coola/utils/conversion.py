r"""Contain conversion functions."""

from __future__ import annotations

__all__ = ["to_jsonable"]

import base64
import datetime
import sys
from dataclasses import fields, is_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def to_jsonable(data: Any) -> Any:
    r"""Convert a single object to a JSON-compatible representation.

    This function converts ``pydantic.BaseModel``, dataclass,
    ``numpy.ndarray``, ``numpy`` scalar, ``torch.Tensor``,
    ``pathlib`` path, ``datetime.date``/``datetime.datetime``/``datetime.time``,
    ``datetime.timedelta``,
    ``decimal.Decimal``, ``uuid.UUID``, ``bytes``/``bytearray`` and
    ``enum.Enum`` and ``range`` objects. Paths, decimals and UUIDs are
    converted to strings, ranges to lists, dates and times to ISO 8601 strings,
    timedeltas to a number of seconds,
    bytes to a Base64 string
    (ASCII) and enum members to their (converted) value.
    Dataclasses are
    converted recursively, field by field, without copying the field
    values (``dataclasses.asdict`` deep-copies them), so a nested
    dataclass or ``BaseModel`` field is converted too. All other object types,
    including containers such as ``list`` or ``dict``, are returned
    unchanged. Non-string ``dict`` keys are not converted, so the result
    is accepted by ``json.dumps`` only if the keys are ``str``, ``int``,
    ``float``, ``bool`` or ``None``; ``NaN`` and infinite floats are
    kept as they are. Sets are not converted (use
    ``coola.nested.convert_to_jsonable``, which converts them to sorted
    lists). To recursively convert every object in a nested data
    structure, use ``coola.nested.convert_to_jsonable`` instead. The
    contract is "accepted by the Python ``json`` module", not "strict
    JSON" (see the notes on the keys and on ``NaN`` above).

    Args:
        data: The object to convert.

    Returns:
        The converted object if it is one of the supported types,
            otherwise the object unchanged.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.utils.conversion import to_jsonable
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> to_jsonable(Point(x=1, y=2))
        {'x': 1, 'y': 2}
        >>> to_jsonable([Point(x=1, y=2)])  # not converted because it is a list
        [Point(x=1, y=2)]

        ```
    """
    for matches, convert in _CONVERTERS:
        if matches(data):
            return convert(data)
    if is_dataclass(data) and not isinstance(data, type):
        return _to_jsonable_recursive({f.name: getattr(data, f.name) for f in fields(data)})
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
    ``torch.Tensor``."""
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


def _to_jsonable_recursive(data: Any) -> Any:
    r"""Recursively apply ``to_jsonable`` to the dict/list structure
    built from the fields of a dataclass.

    The field values (e.g. ``numpy.ndarray``, ``torch.Tensor``,
    ``pydantic.BaseModel``, nested dataclasses) are not converted yet.
    This helper walks the structure and converts each leaf.

    Args:
        data: The dict/list structure to convert.

    Returns:
        The converted object.
    """
    if isinstance(data, dict):
        return {
            key: _to_jsonable_recursive(value)
            for key, value in cast("dict[Any, Any]", data).items()
        }
    if isinstance(data, (list, tuple, set, frozenset)):
        return [_to_jsonable_recursive(value) for value in cast("Iterable[Any]", data)]
    return to_jsonable(data)

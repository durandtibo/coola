r"""Contain conversion functions."""

from __future__ import annotations

__all__ = ["to_jsonable"]

import base64
import datetime
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from coola.utils.imports import (
    is_numpy_available,
    is_pydantic_available,
    is_torch_available,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

if is_pydantic_available():
    from pydantic import BaseModel
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import BaseModel

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


def to_jsonable(data: Any) -> Any:
    r"""Convert a single object to a JSON-compatible representation.

    This function converts ``pydantic.BaseModel``, dataclass,
    ``numpy.ndarray``, ``numpy`` scalar, ``torch.Tensor``,
    ``pathlib`` path, ``datetime.date``/``datetime.datetime``,
    ``decimal.Decimal``, ``uuid.UUID``, ``bytes``/``bytearray`` and
    ``enum.Enum`` objects. Paths, decimals and UUIDs are converted to
    strings, dates to ISO 8601 strings, bytes to a Base64 string
    (ASCII) and enum members to their (converted) value.
    Dataclasses are
    converted recursively because ``dataclasses.asdict`` can produce
    nested dataclass or ``BaseModel`` fields. All other object types,
    including containers such as ``list`` or ``dict``, are returned
    unchanged. To recursively convert every object in a nested data
    structure, use ``coola.nested.convert_to_jsonable`` instead.

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
    for types, convert in _CONVERTERS:
        if isinstance(data, types):
            return convert(data)
    if is_dataclass(data) and not isinstance(data, type):
        return _to_jsonable_recursive(asdict(data))
    return data


# (types, converter) pairs, checked in order by ``to_jsonable``.
_CONVERTERS: tuple[tuple[tuple[type, ...], Callable[[Any], Any]], ...] = (
    ((BaseModel,), lambda data: data.model_dump(mode="json")),
    (
        (np.ndarray, np.generic, torch.Tensor) if is_numpy_available() else (torch.Tensor,),
        lambda data: data.tolist(),
    ),
    ((PurePath, Decimal, UUID), str),
    ((bytes, bytearray), lambda data: base64.b64encode(data).decode("ascii")),
    ((datetime.date,), lambda data: data.isoformat()),
    ((Enum,), lambda data: to_jsonable(data.value)),
)


def _to_jsonable_recursive(data: Any) -> Any:
    r"""Recursively apply ``to_jsonable`` to the dict/list structure
    produced by ``dataclasses.asdict``.

    ``dataclasses.asdict`` recurses into nested dataclasses, ``list``, and
    ``dict`` fields but leaves other field values (e.g. ``numpy.ndarray``,
    ``torch.Tensor``, ``pydantic.BaseModel``) unconverted. This helper walks
    the resulting structure and converts each leaf.

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

r"""Contain conversion functions."""

from __future__ import annotations

__all__ = ["to_json"]

from dataclasses import asdict, is_dataclass
from typing import Any

from coola.utils.imports import is_pydantic_available

if is_pydantic_available():
    from pydantic import BaseModel
else:  # pragma: no cover
    from coola.utils.fallback.pydantic import BaseModel


def to_json(data: Any) -> Any:
    r"""Convert a single object to a JSON-compatible representation.

    This function only converts ``pydantic.BaseModel`` and dataclass
    objects. Dataclasses are converted recursively because
    ``dataclasses.asdict`` can produce nested dataclass or
    ``BaseModel`` fields. All other object types, including
    containers such as ``list`` or ``dict``, are returned unchanged.
    To recursively convert every object in a nested data structure,
    use ``coola.nested.convert_to_json`` instead.

    Args:
        data: The object to convert.

    Returns:
        The converted object if it is a ``pydantic.BaseModel`` or a
            dataclass, otherwise the object unchanged.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.utils.conversion import to_json
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> to_json(Point(x=1, y=2))
        {'x': 1, 'y': 2}
        >>> to_json([Point(x=1, y=2)])  # not converted because it is a list
        [Point(x=1, y=2)]

        ```
    """
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if is_dataclass(data) and not isinstance(data, type):
        return to_json(asdict(data))
    return data

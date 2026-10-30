r"""Define the transformers that convert nested data to JSON data.

The transformers describe the shape of one type with the ``expand``
contract of ``coola.recursive``. The conversion is the third shape: its
result is converted in turn by the engine, so nothing is returned
unchecked.
"""

from __future__ import annotations

__all__ = [
    "ConvertTransformer",
    "DataclassTransformer",
    "MappingTransformer",
    "SequenceTransformer",
    "SetTransformer",
]

import dataclasses
import json
from typing import TYPE_CHECKING, Any, TypeVar

from coola.display import InlineDisplayMixin
from coola.recursive import BaseTransformer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence, Set as AbstractSet

    from coola.recursive.base import Expansion

T = TypeVar("T")


class ConvertTransformer(InlineDisplayMixin, BaseTransformer[T]):
    r"""Transformer that replaces the data by ``func(data)``, which is
    converted in turn.

    The result of ``func`` is converted by the engine like any other
    value, until only JSON scalars and containers remain. A function
    does not call the registry and does not convert its own result: it
    only maps a value to a simpler one (a path to a ``str``, an enum
    member to its value, an array to a list).

    Args:
        func: The function that maps the data to the value to convert.
            Its result must not be of the type of the data, because it
            would be converted by this transformer again, without end.

    Raises:
        TypeError: From ``expand``, if ``func`` returns a value of the
            type of the data.

    Example:
        ```pycon
        >>> from pathlib import PurePosixPath
        >>> from coola.jsonable import ConvertTransformer
        >>> transformer = ConvertTransformer(str)
        >>> children, rebuild = transformer.expand(PurePosixPath("a/b"))
        >>> children
        ['a/b']
        >>> rebuild(children)
        'a/b'

        ```
    """

    def __init__(self, func: Callable[[T], Any]) -> None:
        self._func = func

    def _display_fields(self) -> dict[str, Any]:
        return {"func": self._func}

    def expand(self, data: T) -> Expansion | None:
        converted = self._func(data)
        if type(converted) is type(data):
            msg = (
                f"The converter of {type(data).__qualname__} returned a value of the same "
                f"type: it would be converted without end"
            )
            raise TypeError(msg)
        return [converted], _first


class SequenceTransformer(InlineDisplayMixin, BaseTransformer["Sequence[Any]"]):
    r"""Transformer for sequences (``list``, ``tuple``, ``deque``,
    ``range``, ...) that rebuilds a ``list``.

    The elements are the children, so they are converted like the other
    values.

    Example:
        ```pycon
        >>> from coola.jsonable import SequenceTransformer
        >>> transformer = SequenceTransformer()
        >>> children, rebuild = transformer.expand((1, 2, 3))
        >>> children
        [1, 2, 3]
        >>> rebuild(["1", "2", "3"])
        ['1', '2', '3']

        ```
    """

    def expand(self, data: Sequence[Any]) -> Expansion | None:
        return list(data), _identity


class SetTransformer(InlineDisplayMixin, BaseTransformer["AbstractSet[Any]"]):
    r"""Transformer for sets that rebuilds a sorted ``list``.

    The elements are the children, so they are converted like the other
    values, and the converted elements are collected in a list sorted
    with ``_sorted_elements``. The sort does not depend on the
    iteration order of the set, so the output is the same in every
    process.

    Example:
        ```pycon
        >>> from coola.jsonable import SetTransformer
        >>> transformer = SetTransformer()
        >>> children, rebuild = transformer.expand({3, 1, 2})
        >>> rebuild([3, 1, 2])
        [1, 2, 3]

        ```
    """

    def expand(self, data: AbstractSet[Any]) -> Expansion | None:
        return list(data), _sorted_elements


class MappingTransformer(InlineDisplayMixin, BaseTransformer["Mapping[Any, Any]"]):
    r"""Transformer for mappings that rebuilds a ``dict`` with ``str``
    keys.

    The keys and the values are the children, so a key is converted by
    the registry like any other value (a ``Path`` gives a ``str``). Then
    a ``str`` key is kept, and a ``bool``, ``int``, ``float`` or
    ``None`` key is written like ``json.dumps`` writes it (``"true"``,
    ``"1"``, ``"1.5"``, ``"null"``). The order of the mapping is kept.

    Raises:
        TypeError: From the function that rebuilds the dictionary, if
            a converted key is not a ``str``, ``bool``, ``int``,
            ``float`` or ``None`` (a tuple gives a list).
        ValueError: From the function that rebuilds the dictionary, if
            two keys give the same string.

    Example:
        ```pycon
        >>> from coola.jsonable import MappingTransformer
        >>> transformer = MappingTransformer()
        >>> children, rebuild = transformer.expand({"a": 1, 2: 3})
        >>> children
        ['a', 2, 1, 3]
        >>> rebuild(children)
        {'a': 1, '2': 3}

        ```
    """

    def expand(self, data: Mapping[Any, Any]) -> Expansion | None:
        return [*data.keys(), *data.values()], _to_dict


class DataclassTransformer(InlineDisplayMixin, BaseTransformer[Any]):
    r"""Transformer for dataclass instances that rebuilds a ``dict``.

    The values of the fields are the children, so they are converted
    like the other values, and the converted values are collected in a
    ``dict`` that maps the names of the fields to them. The fields are
    read one by one: ``dataclasses.asdict`` is not used because it
    deep-copies the values.

    Example:
        ```pycon
        >>> from dataclasses import dataclass
        >>> from coola.jsonable import DataclassTransformer
        >>> @dataclass
        ... class Point:
        ...     x: int
        ...     y: int
        ...
        >>> children, rebuild = DataclassTransformer().expand(Point(1, 2))
        >>> children
        [1, 2]
        >>> rebuild(["1", "2"])
        {'x': '1', 'y': '2'}

        ```
    """

    def expand(self, data: Any) -> Expansion | None:
        names = [field.name for field in dataclasses.fields(data)]
        return [getattr(data, name) for name in names], lambda items: dict(
            zip(names, items, strict=True)
        )


def _first(items: list[Any]) -> Any:
    r"""Return the first item.

    Args:
        items: The list with the converted value.

    Returns:
        The first item.
    """
    return items[0]


def _identity(items: list[Any]) -> list[Any]:
    r"""Return the list of the converted children.

    Args:
        items: The converted children, in a new list.

    Returns:
        The same list.
    """
    return items


def _to_key(key: Any) -> str:
    r"""Convert a converted key to a JSON key.

    Args:
        key: The key, already converted by the registry.

    Returns:
        The key as ``json.dumps`` writes it.

    Raises:
        TypeError: If the key is not a ``str``, ``bool``, ``int``,
            ``float`` or ``None``.
    """
    if isinstance(key, str):
        return key
    if key is None or isinstance(key, (bool, int, float)):
        return json.dumps(key)
    msg = (
        f"A JSON key must be a str, bool, int, float or None after the conversion, "
        f"but received a key of type {type(key).__qualname__}"
    )
    raise TypeError(msg)


def _to_dict(items: list[Any]) -> dict[str, Any]:
    r"""Build a dictionary from the converted keys and values.

    Args:
        items: The converted keys followed by the converted values.

    Returns:
        The dictionary with ``str`` keys.

    Raises:
        TypeError: If a key cannot be a JSON key.
        ValueError: If two keys give the same string.
    """
    size = len(items) // 2
    result: dict[str, Any] = {}
    for key, value in zip(items[:size], items[size:], strict=True):
        json_key = _to_key(key)
        if json_key in result:
            msg = f"Two keys give the same JSON key: {json_key!r}"
            raise ValueError(msg)
        result[json_key] = value
    return result


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

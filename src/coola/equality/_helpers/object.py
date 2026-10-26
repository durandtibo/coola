r"""Implement a handler to check if two objects have the same
attributes."""

from __future__ import annotations

__all__ = ["get_object_attributes"]

import functools
from typing import Any


def get_object_attributes(obj: object) -> dict[str, Any]:
    r"""Get the attributes of an object that define its state.

    The attributes are the entries of ``vars(obj)``, the values of the
    slots (``__slots__``) of its class and of its parent classes (a
    slot that is not set is ignored), the ``args`` of an exception and
    the ``func``, ``args`` and ``keywords`` of a
    ``functools.partial``. A slot or an entry of ``vars`` takes
    precedence over the ``args`` of an exception.

    Args:
        obj: The object.

    Returns:
        The attributes, by name.

    Example:
        ```pycon
        >>> from coola.equality._helpers import get_object_attributes
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x = x
        ...         self.y = y
        ...
        >>> get_object_attributes(Point(1, 2))
        {'x': 1, 'y': 2}
        >>> get_object_attributes(ValueError("a"))
        {'args': ('a',)}

        ```
    """
    attributes: dict[str, Any] = {}
    if isinstance(obj, BaseException):
        attributes["args"] = obj.args
    if isinstance(obj, functools.partial):
        attributes.update(func=obj.func, args=obj.args, keywords=obj.keywords)
    for cls in reversed(type(obj).__mro__):
        slots = cls.__dict__.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in {"__dict__", "__weakref__"} or not hasattr(obj, name):
                continue
            attributes[name] = getattr(obj, name)
    attributes.update(getattr(obj, "__dict__", {}))
    return attributes

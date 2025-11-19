r"""Utilities for introspecting Python objects and retrieving their
fully qualified names."""

from __future__ import annotations

__all__ = ["get_fully_qualified_name"]

from typing import Any


def get_fully_qualified_name(obj: Any) -> str:
    r"""Return the fully qualified name of a Python object.

    Supports functions, classes, methods, and instances.
    For instances, returns the fully qualified class name.

    Args:
        obj: The object whose name is to be computed.

    Returns:
        The fully qualified name.

    Example usage:

    ```pycon

    >>> from coola.utils.introspection import get_fully_qualified_name
    >>> import collections
    >>> get_fully_qualified_name(collections.Counter)
    'collections.Counter'
    >>> class MyClass:
    ...     pass
    ...
    >>> get_fully_qualified_name(MyClass)
    '....MyClass'
    >>> get_fully_qualified_name(map)
    'builtins.map'

    ```
    """
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)

    # If not a function/class/method, fall back to the class
    if qualname is None:
        cls = obj.__class__
        module = getattr(cls, "__module__", None)
        qualname = getattr(cls, "__qualname__", None)

    if module and module != "__main__":
        return f"{module}.{qualname}"

    return qualname

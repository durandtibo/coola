r"""Utilities for introspecting Python objects and retrieving their fully
qualified names.

This module provides functions to retrieve information about Python
objects, including their fully qualified names and checking if objects
are lambda functions.
"""

from __future__ import annotations

__all__ = [
    "check_not_lambda_function",
    "get_all_child_classes",
    "get_fully_qualified_name",
    "is_lambda_function",
]

import inspect
from typing import Any


def check_not_lambda_function(obj: Any) -> None:
    r"""Raise an exception if the object is a lambda function.

    Lambda functions cannot be registered because they cannot be
    reliably serialized or referenced by name.

    Args:
        obj: The object to check. Can be any Python object.

    Raises:
        TypeError: if the object is a lambda function.

    Example:
        ```pycon
        >>> from coola.utils.introspection import check_not_lambda_function
        >>> check_not_lambda_function(1)

        ```
    """
    if is_lambda_function(obj):
        msg = (
            "It is not possible to register a lambda function. "
            "Please use a regular function instead"
        )
        raise TypeError(msg)


def get_all_child_classes(cls: type) -> set[type]:
    r"""Get all the child classes (or subclasses) of a given class.

    Based on: https://stackoverflow.com/a/3862957

    Args:
        cls: The class whose child classes are to be retrieved.

    Returns:
        The set of all the child classes of the given class.

    Example:
        ```pycon
        >>> from coola.utils.introspection import get_all_child_classes
        >>> class Foo:
        ...     pass
        ...
        >>> get_all_child_classes(Foo)
        set()
        >>> class Bar(Foo):
        ...     pass
        ...
        >>> get_all_child_classes(Foo)
        {<class '....Bar'>}

        ```
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_child_classes(c)]
    )


def get_fully_qualified_name(obj: Any) -> str:
    r"""Return the fully qualified name of a Python object.

    This function computes the fully qualified name (module path +
    object name) for various Python objects including functions,
    classes, methods, and instances. For instances, it returns the
    fully qualified name of their class. The format is
    "module.path.ObjectName" (e.g., "collections.Counter").

    Args:
        obj: The object whose fully qualified name is to be computed.
            Can be a class, function, method, or instance.

    Returns:
        The fully qualified name.

    Example:
        ```pycon
        >>> from coola.utils.introspection import get_fully_qualified_name
        >>> import collections
        >>> get_fully_qualified_name(collections.Counter)
        'collections.Counter'
        >>> get_fully_qualified_name(collections.Counter())
        'collections.Counter'
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
        qualname = cls.__qualname__

    if module and module != "__main__":
        return f"{module}.{qualname}"
    return qualname


def is_lambda_function(obj: Any) -> bool:
    r"""Indicate if the object is a lambda function or not.

    This function checks whether a given object is a lambda function
    by examining its type and name. Lambda functions are not allowed
    in factories because they cannot be reliably serialized or
    referenced by name. Adapted from
    https://stackoverflow.com/a/23852434

    Args:
        obj: The object to check. Can be any Python object.

    Returns:
        ``True`` if the input is a lambda function,
            otherwise ``False``

    Example:
        ```pycon
        >>> from coola.utils.introspection import is_lambda_function
        >>> is_lambda_function(lambda value: value + 1)
        True
        >>> def my_function(value: int) -> int:
        ...     return value + 1
        ...
        >>> is_lambda_function(my_function)
        False
        >>> is_lambda_function(1)
        False

        ```
    """
    if not inspect.isfunction(obj):
        return False
    return obj.__name__ == "<lambda>"

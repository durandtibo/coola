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
    "get_type_constructor",
    "is_lambda_function",
    "is_same_type",
    "supports_methods",
]

import inspect
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


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


def supports_methods(obj: Any, *method_names: str) -> bool:
    r"""Indicate whether an object has all the given methods and they are
    callable.

    Args:
        obj: The object to check.
        method_names: The names of the methods to check.

    Returns:
        ``True`` if ``obj`` has every method in ``method_names`` and
            each is callable, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.introspection import supports_methods
        >>> supports_methods(1, "bit_length")
        True
        >>> supports_methods(1, "bit_length", "does_not_exist")
        False

        ```
    """
    return all(hasattr(obj, name) and callable(getattr(obj, name)) for name in method_names)


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


def get_type_constructor(obj: object) -> Callable[..., Any]:
    r"""Get the type of an object as a generic callable.

    Type checkers do not allow calling ``type(obj)`` with arbitrary
    arguments. This helper centralizes the cast needed to rebuild an
    object of the same type from new data.

    Args:
        obj: The object whose type is returned.

    Returns:
        The type of ``obj``, typed as a callable.

    Example:
        ```pycon
        >>> from coola.utils.introspection import get_type_constructor
        >>> get_type_constructor([1, 2])((3, 4))
        [3, 4]

        ```
    """
    return cast("Callable[..., Any]", type(obj))


def is_same_type(value: object, reference: T) -> TypeGuard[T]:
    r"""Indicate if an object has exactly the same type as a reference.

    The types must be identical: an instance of a subclass is not
    considered to have the same type as an instance of its parent
    class. This is the check used by the ``equal`` methods to compare
    two objects of the same class. When it returns ``True``, type
    checkers know that ``value`` has the type of ``reference``, so its
    attributes can be accessed.

    Args:
        value: The object to check. It is the first argument because
            type guards narrow the first argument.
        reference: The object that gives the expected type.

    Returns:
        ``True`` if ``type(value) is type(reference)``, otherwise
            ``False``.

    Example:
        ```pycon
        >>> from coola.utils.introspection import is_same_type
        >>> is_same_type(1, 2)
        True
        >>> is_same_type(1, 2.0)
        False
        >>> is_same_type(True, 1)
        False

        ```
    """
    return type(value) is type(reference)

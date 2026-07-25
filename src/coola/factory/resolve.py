r"""Implement a "universal" factory function."""

from __future__ import annotations

__all__ = ["factory"]

from typing import Any

from coola.factory.instantiation import import_object, instantiate_object


def factory(_target_: str, *args: Any, _init_: str = "__init__", **kwargs: Any) -> Any:
    r"""Instantiate dynamically an object given its configuration.

    This function provides a universal factory that can instantiate
    any class or call any function by its fully qualified name. Unlike
    the AbstractFactory or Registry approaches, this function does not
    require prior registration of classes.

    Args:
        _target_: The fully qualified name of the object (class or
            function) to instantiate, e.g., "collections.Counter" or
            "math.isclose".
        *args: Positional arguments to pass to the class constructor
            or function.
        _init_: The function or method to use to create the object.
            If ``"__init__"`` (default), the object is created by
            calling the constructor. Can also be ``"__new__"`` or the
            name of a class method.
        **kwargs: Keyword arguments to pass to the class constructor
            or function.

    Returns:
        The instantiated object with the given parameters.

    Raises:
        ImportError: if the target cannot be found.
        TypeError: if ``_target_`` is not a string.

    Example:
        ```pycon
        >>> from coola.factory import factory
        >>> factory("collections.Counter", [1, 2, 1, 3])
        Counter({1: 2, 2: 1, 3: 1})

        ```
    """
    try:
        target = import_object(_target_)
    except ImportError as e:
        msg = f"The target object does not exist: {_target_}"
        raise ImportError(msg) from e
    return instantiate_object(target, *args, _init_=_init_, **kwargs)

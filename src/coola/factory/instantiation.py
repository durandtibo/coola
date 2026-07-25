r"""Utilities for instantiating Python objects."""

from __future__ import annotations

__all__ = ["import_object", "instantiate_object"]

import importlib
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


def import_object(object_path: str) -> Any:
    r"""Import an object given its path.

    This function dynamically imports a class, function, or other
    Python object using its fully qualified name. The object path
    should have the structure ``module_path.object_name`` (e.g.,
    "collections.Counter" or "math.isclose").

    Args:
        object_path: The fully qualified path of the object to import.
            Must be a string in the format "module.path.ObjectName".

    Returns:
        The imported object.

    Raises:
        TypeError: if ``object_path`` is not a string.
        ImportError: if ``object_path`` cannot be imported.

    Example:
        ```pycon
        >>> from coola.factory import import_object
        >>> cls = import_object("collections.Counter")
        >>> cls()
        Counter()
        >>> fn = import_object("math.isclose")
        >>> fn(1, 1)
        True
        >>> pi = import_object("math.pi")
        >>> pi
        3.141592653589793
        >>> pkg = import_object("math")
        >>> pkg
        <module 'math' (built-in)>

        ```
    """
    if not isinstance(object_path, str):
        msg = f"`object_path` is not a string: {object_path}"
        raise TypeError(msg)

    # If there's no dot, treat it as a module/package import.
    if "." not in object_path:
        return importlib.import_module(object_path)

    module_name, _, attr = object_path.rpartition(".")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as err:
        msg = f"Module {module_name!r} has no attribute {attr!r}"
        raise ImportError(msg) from err


def instantiate_object(
    obj: Callable | type, *args: Any, _init_: str = "__init__", **kwargs: Any
) -> Any:
    r"""Instantiate dynamically an object from its configuration.

    This function creates an instance of a class or calls a callable
    with the provided arguments. For classes, it supports different
    instantiation methods (constructor, __new__, or class methods).
    For any other callable (function, builtin, ``functools.partial``,
    or an object implementing ``__call__``), it simply calls it with
    the given arguments.

    Args:
        obj: The class to instantiate or the callable to call. Must
            be a class or a callable object.
        *args: Positional arguments to pass to the class constructor
            or callable.
        _init_: The function or method to use to create the object.
            This parameter is ignored if ``obj`` is not a class. For
            classes, if ``"__init__"`` (default), the object is
            created by calling the constructor. Can also be
            ``"__new__"`` or the name of a class method.
        **kwargs: Keyword arguments to pass to the class constructor
            or callable.

    Returns:
        The instantiated object if ``obj`` is a class, otherwise the
            returned value of the callable.

    Raises:
        TypeError: if ``obj`` is not a class or a callable.

    Example:
        ```pycon
        >>> from collections import Counter
        >>> from coola.factory import instantiate_object
        >>> instantiate_object(Counter, [1, 2, 1])
        Counter({1: 2, 2: 1})
        >>> instantiate_object(list, [1, 2, 1])
        [1, 2, 1]

        ```
    """
    if inspect.isclass(obj):
        return _instantiate_class_object(obj, *args, _init_=_init_, **kwargs)
    if callable(obj):
        return obj(*args, **kwargs)
    msg = f"Incorrect type: {obj}. The valid types are class and callable"
    raise TypeError(msg)


def _instantiate_class_object(
    cls: type, *args: Any, _init_: str = "__init__", **kwargs: Any
) -> Any:
    r"""Instantiate an object from its class and some arguments.

    The object can be instantiated by calling the constructor
    ``__init__`` (default) or ``__new__`` or a class method.

    Args:
        cls: The class of the object to instantiate.
        *args: Variable length argument list.
        _init_: The function to use to create the object.
            If ``"__init__"``, the object is created by calling the
            constructor.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The instantiated object.

    Raises:
        TypeError: if it is an abstract class, if it is not possible
            to instantiate the object, or if the object returned by
            ``_init_`` is not an instance of ``cls``.
    """
    if inspect.isabstract(cls):
        msg = f"Cannot instantiate the class {cls} because it is an abstract class."
        raise TypeError(msg)

    if _init_ == "__init__":
        return cls(*args, **kwargs)

    if not hasattr(cls, _init_):
        msg = f"{cls} does not have `{_init_}` attribute"
        raise TypeError(msg)
    init_fn = getattr(cls, _init_)
    if not callable(init_fn):
        msg = f"`{_init_}` attribute of {cls} is not callable"
        raise TypeError(msg)
    obj = init_fn(cls, *args, **kwargs) if _init_ == "__new__" else init_fn(*args, **kwargs)
    if not isinstance(obj, cls):
        msg = (
            f"`{_init_}` of {cls} did not return an instance of {cls} "
            f"(received: {obj!r} of type {type(obj)})"
        )
        raise TypeError(msg)
    return obj

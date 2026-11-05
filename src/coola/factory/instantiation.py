r"""Utilities for instantiating Python objects."""

from __future__ import annotations

__all__ = ["import_object", "instantiate_object"]

import importlib
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def import_object(object_path: object, *, allowed_prefixes: Sequence[str] | None = None) -> Any:
    r"""Import an object given its path.

    This function dynamically imports a class, function, or other
    Python object using its fully qualified name. The object path
    should have the structure ``module_path.object_name`` (e.g.,
    "collections.Counter" or "math.isclose"). The object can also be
    an attribute of a class (``"collections.Counter.fromkeys"``): the
    longest prefix that is a module is imported and the rest of the path
    is read with ``getattr``.

    Args:
        object_path: The fully qualified path of the object to import.
            Must be a string in the format "module.path.ObjectName".
        allowed_prefixes: An optional allow-list of module prefixes.
            If provided, ``object_path`` must be equal to a prefix or
            start with ``prefix + "."``, otherwise an ``ImportError``
            is raised *before* anything is imported. A string is
            treated as a single prefix.

    Returns:
        The imported object.

    Raises:
        TypeError: if ``object_path`` is not a string.
        ImportError: if ``object_path`` is empty or has an empty module
            name (``""``, ``".Counter"``), is not allowed by
            ``allowed_prefixes``, or if its module or attribute does not
            exist (``"The target object does not exist: ..."``). A
            dependency of the target module that is not installed keeps
            its own ``ModuleNotFoundError``.

    Note:
        ``pkgutil.resolve_name`` is a different function:
        it takes a path with a colon (``"package.module:name"``) and is
        used by the lazy registries. This function takes a dotted path
        (``"package.module.name"``), checks ``allowed_prefixes`` and
        reports a missing target with one ``ImportError``.

    Security:
        By default, this function imports arbitrary module code with
        no restriction on which module or attribute it may reference:
        importing a module executes its top-level code. Use
        ``allowed_prefixes`` to restrict it. Only pass an
        ``object_path`` that comes from a trusted source. Never
        resolve an ``object_path`` derived from untrusted input, such
        as a third-party-uploaded config file or a network payload, as
        this is a remote-code-execution vector.

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

    if isinstance(allowed_prefixes, str):
        # A string is a single prefix, not a sequence of one-character prefixes.
        allowed_prefixes = (allowed_prefixes,)
    if allowed_prefixes is not None and not any(
        object_path == prefix or object_path.startswith(prefix + ".") for prefix in allowed_prefixes
    ):
        msg = f"Object path {object_path!r} is not allowed. Allowed prefixes: {list(allowed_prefixes)}"
        raise ImportError(msg)

    # An empty attribute name ("collections.") is reported below with the module name.
    if not object_path or not all(object_path.split(".")[:-1]):
        msg = f"Invalid object path {object_path!r}: empty module name"
        raise ImportError(msg)

    return _resolve(object_path)


def _resolve(object_path: str) -> Any:
    r"""Import the object of a path.

    The module is the longest prefix of the path that can be imported,
    and the rest of the path is read with ``getattr``, so an attribute
    of a class (``"module.Class.attribute"``) is found.

    Args:
        object_path: The path of the object.

    Returns:
        The object.

    Raises:
        ImportError: If the object does not exist. The error is the one
            of the longest prefix.
    """
    parts = object_path.split(".")
    failures: list[tuple[str | None, BaseException | None]] = []
    for end in range(max(len(parts) - 1, 1), 0, -1):
        try:
            obj = _import_module(".".join(parts[:end]), object_path)
            for attr in parts[end:]:
                obj = getattr(obj, attr)
        except _TargetNotFoundError as err:
            failures.append((err.name, err.__cause__))
        except AttributeError as err:
            failures.append((object_path, err))
        else:
            return obj
    name, cause = failures[0]
    msg = f"The target object does not exist: {object_path}"
    raise ImportError(msg, name=name) from cause


class _TargetNotFoundError(ImportError):
    r"""Raised when the module of the target of ``import_object`` does
    not exist."""


def _import_module(module_name: str, object_path: str) -> Any:
    r"""Import a module for ``import_object``.

    Args:
        module_name: The name of the module to import.
        object_path: The path of the object that is imported, used in
            the error.

    Returns:
        The module.

    Raises:
        _TargetNotFoundError: If the module (or one of its parents)
            does not exist. It is an ``ImportError``. A dependency of
            the module that is not installed keeps its own
            ``ModuleNotFoundError``.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        # Only the module of the target (or one of its parents) is "missing". A dependency of the
        # target module that is not installed keeps its own error.
        if err.name and (object_path == err.name or object_path.startswith(err.name + ".")):
            msg = f"The target object does not exist: {object_path}"
            raise _TargetNotFoundError(msg, name=err.name) from err
        raise


def instantiate_object(
    obj: Callable[..., Any] | type, /, *args: Any, _init_: str = "__init__", **kwargs: Any
) -> Any:
    r"""Instantiate dynamically an object from its configuration.

    This function creates an instance of a class or calls a callable
    with the provided arguments. For classes, it supports different
    instantiation methods (constructor or public class methods).
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
            created by calling the constructor (``__new__`` and
            ``__init__``). Can also be the name of a public class
            method or static method that returns an instance of the
            class. A name that starts with ``_`` is refused.
        **kwargs: Keyword arguments to pass to the class constructor
            or callable. Any name is accepted, including ``obj`` and
            ``cls``: ``obj`` is positional-only.

    Returns:
        The instantiated object if ``obj`` is a class, otherwise the
            returned value of the callable.

    Raises:
        TypeError: if ``obj`` is not a class or a callable, if it is an
            abstract class, if ``_init_`` is not a public method name
            (it is not a string or it starts with ``_``), if the class
            does not have a callable ``_init_`` attribute, or if
            ``_init_`` does not return an instance of ``obj``.

    Security:
        ``obj`` is instantiated or called with the given arguments with
        no allowlist or restriction on what it may do: an arbitrary
        class or callable will be invoked as-is. Only pass an ``obj``
        (and ``_init_``) that come from a trusted source. Never
        instantiate an object whose class/callable or arguments are
        derived from untrusted input, as this is a remote-code-execution
        vector.

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
    if not callable(obj):
        msg = f"Incorrect type: {obj}. The valid types are class and callable"
        raise TypeError(msg)
    if not isinstance(obj, type):
        return obj(*args, **kwargs)
    if inspect.isabstract(obj):
        msg = f"Cannot instantiate the class {obj} because it is an abstract class."
        raise TypeError(msg)
    if _init_ == "__init__":
        return obj(*args, **kwargs)
    if not isinstance(_init_, str) or _init_.startswith("_"):
        msg = (
            "`_init_` must be '__init__' or the name of a public class method "
            f"(received: {_init_!r})"
        )
        raise TypeError(msg)
    method = getattr(obj, _init_, None)
    if not callable(method):
        msg = f"{obj} does not have a callable `{_init_}` attribute"
        raise TypeError(msg)
    result = method(*args, **kwargs)
    if not isinstance(result, obj):
        msg = (
            f"`{_init_}` of {obj} did not return an instance of {obj} "
            f"(received: {result!r} of type {type(result)})"
        )
        raise TypeError(msg)
    return result

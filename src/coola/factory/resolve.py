r"""Implement a "universal" factory function."""

from __future__ import annotations

__all__ = ["factory", "resolve_object"]

import logging
from typing import Any, TypeVar

from coola.factory.constants import OBJECT_TARGET
from coola.factory.instantiation import import_object, instantiate_object

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


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


def resolve_object(obj: T | dict[str, Any], cls: type[T] = object) -> T:
    """Resolve an instance of ``cls`` from an existing object or a
    configuration dictionary.

    If ``obj`` is already an instance of ``cls`` it is returned
    as-is.  If it is a :class:`dict`, it is treated as a factory
    configuration and instantiated via :func:`coola.factory.factory`.

    Note:
        Any :class:`dict` (including instances of ``dict``
        subclasses, e.g. ``Counter`` or ``OrderedDict``) is always
        treated as a factory configuration, even when it is already
        a valid instance of ``cls``. Do not use this function to
        resolve objects whose expected type is itself a ``dict``
        subclass.

    Args:
        obj: Either a fully configured instance of ``cls``, or a
            :class:`dict` containing a factory specification (must
            include a ``"_target_"`` key pointing to the
            fully-qualified class name).
        cls: The expected type. Used to validate the resolved object,
            whether ``obj`` was already an instance or was built from
            a configuration dictionary. Defaults to :class:`object`,
            which accepts any resolved value without validation.

    Returns:
        A configured instance of ``cls``.

    Raises:
        TypeError: If ``obj`` is a :class:`dict` missing the
            ``"_target_"`` key, or if the resolved object is not an
            instance of ``cls``.

    Example:
        ```pycon
        >>> from datetime import date
        >>> from coola.factory import resolve_object
        >>> # From an existing instance:
        >>> d = resolve_object(date(2020, 1, 1), cls=date)
        >>> # From a configuration dictionary:
        >>> d = resolve_object(
        ...     {"_target_": "datetime.date", "year": 2020, "month": 1, "day": 1}, cls=date
        ... )

        ```
    """
    cls_name = cls.__qualname__
    if isinstance(obj, dict):
        if OBJECT_TARGET not in obj:
            msg = (
                f"Cannot resolve a {cls_name} instance from the configuration because it is "
                f"missing the `{OBJECT_TARGET}` key (received: {obj})"
            )
            raise TypeError(msg)
        logger.info("Initializing a %s instance from its configuration...", cls_name)
        obj = factory(**obj)
    if not isinstance(obj, cls):
        msg = f"Received object is not a {cls_name} instance (received: {type(obj)})"
        raise TypeError(msg)
    return obj

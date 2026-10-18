r"""Implement a "universal" factory function."""

from __future__ import annotations

__all__ = ["factory", "resolve_object"]

import logging
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from coola.factory.constants import OBJECT_INIT, OBJECT_TARGET
from coola.factory.instantiation import import_object, instantiate_object

if TYPE_CHECKING:
    from collections.abc import Sequence

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def factory(_target_: str, *args: Any, **kwargs: Any) -> Any:
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
        **kwargs: Keyword arguments to pass to the class constructor
            or function. The special ``OBJECT_INIT`` (``"_init_"``)
            key controls the function or method used to create the
            object. If ``"__init__"`` (default), the object is
            created by calling the constructor. Can also be
            ``"__new__"`` or the name of a class method. The special
            ``"_allowed_prefixes_"`` key is an optional allow-list
            of module prefixes that ``_target_`` must match (see
            ``import_object``).

    Returns:
        The instantiated object with the given parameters.

    Raises:
        ImportError: if the target cannot be found.
        TypeError: if ``_target_`` is not a string.

    Security:
        ``_target_`` is imported and called with no allowlist or
        restriction on which module or class it may reference: this
        function will import arbitrary module code and instantiate or
        call arbitrary objects from a dotted path string. Only pass a
        ``_target_`` that comes from a trusted source (e.g. code you
        wrote, or a configuration file you control). Never resolve a
        ``_target_`` derived from untrusted input, such as a
        third-party-uploaded config file or a network payload, as this
        is a remote-code-execution vector.

    Example:
        ```pycon
        >>> from coola.factory import factory
        >>> factory("collections.Counter", [1, 2, 1, 3])
        Counter({1: 2, 2: 1, 3: 1})

        ```
    """
    try:
        target = import_object(_target_, allowed_prefixes=kwargs.pop("_allowed_prefixes_", None))
    except ImportError as e:
        msg = f"The target object does not exist: {_target_}"
        raise ImportError(msg) from e
    init = kwargs.pop(OBJECT_INIT, "__init__")
    return instantiate_object(target, *args, _init_=init, **kwargs)


@overload
def resolve_object(
    obj: T | dict[str, Any], *, allowed_prefixes: Sequence[str] | None = None
) -> T: ...


@overload
def resolve_object(
    obj: T | dict[str, Any], cls: type[T], *, allowed_prefixes: Sequence[str] | None = None
) -> T: ...


def resolve_object(
    obj: Any, cls: type = object, *, allowed_prefixes: Sequence[str] | None = None
) -> Any:
    """Resolve an instance of ``cls`` from an existing object or a
    configuration dictionary.

    If ``obj`` is already an instance of ``cls`` it is returned
    as-is. Otherwise, if it is a :class:`dict`, it is treated as a
    factory configuration and instantiated via
    :func:`coola.factory.factory`.

    Note:
        If ``cls`` is itself a :class:`dict` subclass (e.g.
        ``Counter`` or ``OrderedDict``), a ``dict`` (or ``dict``
        subclass) instance that is already a valid instance of
        ``cls`` is returned as-is rather than being treated as a
        factory configuration. Any other :class:`dict` is treated as
        a factory configuration, including a plain :class:`dict`
        used to describe how to build a ``dict``-subclass instance,
        and a ``dict`` subclass instance that is not itself a valid
        ``cls`` instance (e.g. a ``Counter`` when ``cls`` is
        ``OrderedDict``).

    Args:
        obj: Either a fully configured instance of ``cls``, or a
            :class:`dict` containing a factory specification (must
            include a ``"_target_"`` key pointing to the
            fully-qualified class name).
        cls: The expected type. Used to validate the resolved object,
            whether ``obj`` was already an instance or was built from
            a configuration dictionary. Defaults to :class:`object`,
            which accepts any resolved value without validation.
        allowed_prefixes: An optional allow-list of module prefixes
            that the ``"_target_"`` of the configuration must match
            (see ``import_object``): ``"mypackage"`` allows
            ``"mypackage"`` and ``"mypackage.sub.Class"`` but not
            ``"mypackage2.Class"``. If ``None`` (default), any
            ``"_target_"`` is allowed. When it is set, it replaces the
            ``"_allowed_prefixes_"`` key of the configuration, so
            a configuration cannot widen it.

    Returns:
        A configured instance of ``cls``.

    Raises:
        TypeError: If ``obj`` is a :class:`dict` missing the
            ``"_target_"`` key, or if the resolved object is not an
            instance of ``cls``.
        ImportError: If the ``"_target_"`` does not match
            ``allowed_prefixes``, or if it cannot be found.

    Security:
        When ``obj`` is a :class:`dict`, its ``"_target_"`` value is
        passed to :func:`coola.factory.factory`, which imports and
        instantiates arbitrary objects from a dotted path string with
        no allowlist by default. Only call this function with a
        configuration dictionary that comes from a trusted source;
        never resolve a ``_target_`` derived from untrusted input, as
        this is a remote-code-execution vector. If the configuration
        cannot be fully trusted, pass ``allowed_prefixes`` to restrict
        the ``"_target_"`` to the modules you expect, and check the
        arguments of the allowed objects too: an allow-list limits
        *what* can be called, not the arguments it is called with.

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
    cls_is_dict_subclass = issubclass(cls, dict)
    if isinstance(obj, dict) and not (cls_is_dict_subclass and isinstance(obj, cls)):
        if OBJECT_TARGET not in obj:
            msg = (
                f"Cannot resolve a {cls_name} instance from the configuration because it is "
                f"missing the `{OBJECT_TARGET}` key (received: {obj})"
            )
            raise TypeError(msg)
        logger.info("Initializing a %s instance from its configuration...", cls_name)
        config = cast("dict[str, Any]", obj)
        if allowed_prefixes is not None:
            # The caller's allow-list wins over the one of the configuration.
            config = {**config, "_allowed_prefixes_": allowed_prefixes}
        obj = factory(**config)
    if not isinstance(obj, cls):
        msg = (
            f"Received object is not a {cls_name} instance (received: {type(cast('object', obj))})"
        )
        raise TypeError(msg)
    return obj

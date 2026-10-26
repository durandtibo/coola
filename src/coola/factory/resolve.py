r"""Implement a "universal" factory function."""

from __future__ import annotations

__all__ = ["factory", "resolve_object"]

import logging
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

from coola.factory.constants import OBJECT_INIT, OBJECT_PREFIXES, OBJECT_TARGET
from coola.factory.instantiation import import_object, instantiate_object
from coola.factory.target import builds

if TYPE_CHECKING:
    from collections.abc import Sequence

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


def factory(
    _target_: str,
    *args: Any,
    _init_: str = "__init__",
    _allowed_prefixes_: Sequence[str] | None = None,
    **kwargs: Any,
) -> Any:
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
        _init_: The function or method used to create the object.
            If ``"__init__"`` (default), the object is created by
            calling the constructor. Can also be the name of a public
            class method or static method that returns an instance of
            the class (a name that starts with ``_``, such as
            ``"__new__"``, is refused).
        _allowed_prefixes_: An optional allow-list of module prefixes
            that ``_target_`` must match (see ``import_object``). It is
            keyword-only, and it is the choice of the code that calls
            this function: whoever unpacks untrusted data into
            ``factory`` owns it. When it is set, an argument that is
            a configuration is refused.
        **kwargs: Keyword arguments to pass to the class constructor
            or function. Any name is accepted, including ``obj`` and
            ``cls``.

    Returns:
        The instantiated object with the given parameters.

    Raises:
        ImportError: if the target cannot be found, is not allowed by
            ``_allowed_prefixes_``, or if importing its module fails
            (the original error is then raised unchanged).
        TypeError: if ``_target_`` is not a string.
        ValueError: if ``_allowed_prefixes_`` is set and an argument
            is a ``dict`` with a ``_target_`` key (a nested
            configuration): the target could resolve it without the
            allow-list. Resolve it first, with the same list.

    Note:
        ``_target_`` is resolved at the top level only: a nested
        dictionary with a ``_target_`` key is passed to the target
        unchanged, it is not instantiated. Resolve the nested
        configurations with ``resolve_object`` and pass the results.

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
    if _allowed_prefixes_ is not None:
        _refuse_nested_configurations(args, kwargs)
    target = import_object(_target_, allowed_prefixes=_allowed_prefixes_)
    return instantiate_object(target, *args, _init_=_init_, **kwargs)


def _refuse_nested_configurations(args: Sequence[Any], kwargs: dict[str, Any]) -> None:
    r"""Refuse an argument that is a configuration (a ``dict`` with a
    ``_target_`` key).

    Only the top level of the arguments is scanned. With an allow-list,
    the target could resolve such an argument without the allow-list.
    """
    for value in (*args, *kwargs.values()):
        if isinstance(value, dict) and OBJECT_TARGET in value:
            msg = (
                f"An argument is a configuration (a dict with the `{OBJECT_TARGET}` key), which "
                "is refused when `allowed_prefixes` is set: the target could resolve it without "
                "the allow-list. Resolve it first, with the same allow-list, and pass the result "
                f"(received: {value})"
            )
            raise ValueError(msg)


def _check_target(
    config: dict[str, Any],
    cls: type | tuple[type, ...],
    cls_name: str,
    *,
    strict: bool,
) -> None:
    r"""Refuse a configuration whose target cannot build an instance of
    ``cls``, before the target is called.

    A class that is not a subclass of ``cls`` is refused. A function is
    refused if its return type hint is a class that is not a subclass
    of ``cls``; a function without a usable type hint is accepted
    (the result is checked after the call) unless ``strict`` is
    ``True``, in which case it is refused. A configuration with an
    alternative constructor (``_init_``) is not checked.
    """
    if cls is object or OBJECT_INIT in config:
        return
    target = import_object(config[OBJECT_TARGET], allowed_prefixes=config.get(OBJECT_PREFIXES))
    built = builds(target, cls)
    if built is False:
        msg = (
            f"Received object is not a {cls_name} instance (the target "
            f"{config[OBJECT_TARGET]!r} does not build one: it was not called)"
        )
        raise TypeError(msg)
    if built is None and strict:
        msg = (
            f"Cannot check that the target {config[OBJECT_TARGET]!r} builds a {cls_name} "
            "instance because it is not a class and has no return type hint that is a class: "
            "it was not called"
        )
        raise TypeError(msg)


@overload
def resolve_object(
    obj: T | dict[str, Any], *, allowed_prefixes: Sequence[str] | None = None, strict: bool = False
) -> T: ...


@overload
def resolve_object(
    obj: T | dict[str, Any],
    cls: type[T],
    *,
    allowed_prefixes: Sequence[str] | None = None,
    strict: bool = False,
) -> T: ...


@overload
def resolve_object(
    obj: Any,
    cls: tuple[type, ...],
    *,
    allowed_prefixes: Sequence[str] | None = None,
    strict: bool = False,
) -> Any: ...


def resolve_object(
    obj: Any,
    cls: type | tuple[type, ...] = object,
    *,
    allowed_prefixes: Sequence[str] | None = None,
    strict: bool = False,
) -> Any:
    """Resolve an instance of ``cls`` from an existing object or a
    configuration dictionary.

    If ``obj`` is already an instance of ``cls`` it is returned
    as-is. Otherwise, if it is a ``dict``, it is treated as a
    factory configuration and instantiated via
    ``coola.factory.factory``.

    Note:
        ``"_target_"`` is resolved at the top level only: a nested
        dictionary with a ``"_target_"`` key is passed to the target
        unchanged, it is not instantiated. With ``allowed_prefixes``
        it is refused with a ``ValueError``. Resolve the nested
        configurations with ``resolve_object`` (with the same
        ``allowed_prefixes``) and pass the results.

    Note:
        If ``cls`` is itself a ``dict`` subclass (e.g.
        ``Counter`` or ``OrderedDict``), a ``dict`` (or ``dict``
        subclass) instance that is already a valid instance of
        ``cls`` is returned as-is rather than being treated as a
        factory configuration. Any other ``dict`` is treated as
        a factory configuration, including a plain ``dict``
        used to describe how to build a ``dict``-subclass instance,
        and a ``dict`` subclass instance that is not itself a valid
        ``cls`` instance (e.g. a ``Counter`` when ``cls`` is
        ``OrderedDict``).

    Args:
        obj: Either a fully configured instance of ``cls``, or a
            ``dict`` containing a factory specification (must
            include a ``"_target_"`` key pointing to the
            fully-qualified class name).
        cls: The expected type, or a tuple of types (as accepted by
            ``isinstance``). Used to validate the resolved object,
            whether ``obj`` was already an instance or was built from
            a configuration dictionary. Defaults to ``object``,
            which accepts any resolved value without validation.
        allowed_prefixes: An optional allow-list of module prefixes
            that the ``"_target_"`` of the configuration must match
            (see ``import_object``): ``"mypackage"`` allows
            ``"mypackage"`` and ``"mypackage.sub.Class"`` but not
            ``"mypackage2.Class"``. If ``None`` (default), any
            ``"_target_"`` is allowed. The ``"_allowed_prefixes_"`` key
            of a configuration is ignored: the data cannot set, widen
            or narrow the list. When it is set, an argument that is
            a configuration is refused.
        strict: If ``True``, a ``"_target_"`` that is not a class and
            has no usable return type hint (a function without
            annotation, a built-in function) is refused before it is
            called, because the type of its result cannot be checked
            beforehand. If ``False`` (default), it is called and the
            result is checked afterwards.

    Returns:
        A configured instance of ``cls``.

    Raises:
        TypeError: If ``obj`` is a ``dict`` missing the
            ``"_target_"`` key, if the ``"_target_"`` is a class (or a
            function whose return type hint is a class) that is not a
            subclass of ``cls`` -- it is then not called --, or if
            the resolved object is not an instance of ``cls``.
        ImportError: If the ``"_target_"`` does not match
            ``allowed_prefixes``, or if it cannot be found.
        ValueError: If ``allowed_prefixes`` is set and an argument of
            the configuration is itself a configuration (a ``dict``
            with a ``"_target_"`` key).

    Security:
        When ``obj`` is a ``dict``, its ``"_target_"`` value is
        passed to ``coola.factory.factory``, which imports and
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
    classes = cls if isinstance(cls, tuple) else (cls,)
    cls_name = " | ".join(c.__qualname__ for c in classes)
    cls_is_dict_subclass = any(issubclass(c, dict) for c in classes)
    if isinstance(obj, dict) and not (cls_is_dict_subclass and isinstance(obj, cls)):
        if OBJECT_TARGET not in obj:
            msg = (
                f"Cannot resolve a {cls_name} instance from the configuration because it is "
                f"missing the `{OBJECT_TARGET}` key (received: {obj})"
            )
            raise TypeError(msg)
        logger.info("Initializing a %s instance from its configuration...", cls_name)
        config: dict[str, Any] = dict(cast("dict[str, Any]", obj))
        # The policy comes from the caller: the key of the configuration is ignored.
        config.pop(OBJECT_PREFIXES, None)
        if allowed_prefixes is not None:
            _refuse_nested_configurations((), config)
            config[OBJECT_PREFIXES] = allowed_prefixes
        _check_target(config, cls, cls_name, strict=strict)
        obj = factory(**config)
    if not isinstance(obj, cls):
        msg = (
            f"Received object is not a {cls_name} instance (received: {type(cast('object', obj))})"
        )
        raise TypeError(msg)
    return obj

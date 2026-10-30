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

_RESERVED = (OBJECT_TARGET, OBJECT_INIT, OBJECT_PREFIXES)


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
        is a remote-code-execution vector. ``_allowed_prefixes_`` is a
        filter on a path, not a sandbox: it limits *what* is called,
        not the arguments. A prefix that contains a function that
        imports or resolves objects from its arguments (any
        ``resolve_*``, ``import_object``, ``importlib``, ``pickle``,
        ``eval``, ``os.system``, ...) makes the list worthless: never
        allow ``"coola"`` or ``"coola.factory"``. The path is checked,
        not the module that defines the object, so a re-export is
        allowed by the prefix of the re-exporting module.

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

    Args:
        args: The positional arguments of the target.
        kwargs: The keyword arguments of the target.

    Raises:
        ValueError: If an argument is a configuration.
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


def _build(
    config: dict[str, Any],
    cls: type | tuple[type, ...],
    cls_name: str,
    *,
    allowed_prefixes: Sequence[str] | None,
    strict: bool,
) -> Any:
    r"""Build the object of a configuration.

    The target is imported once. It is refused before it is called if it
    is known not to build an instance of ``cls`` (or if it is not known
    and ``strict`` is ``True``). The configuration is not modified.

    Args:
        config: The configuration, with the key ``_target_``.
        cls: The expected class, or a tuple of classes.
        cls_name: The name of the expected class(es), used in the error
            messages.
        allowed_prefixes: The allow-list of module prefixes of the
            target, or ``None`` to allow any target.
        strict: If ``True``, a target that is not known to build an
            instance of ``cls`` is refused.

    Returns:
        The built object.

    Raises:
        TypeError: If the target does not build an instance of ``cls``,
            or if ``strict`` is ``True`` and this cannot be checked.
        ValueError: If ``allowed_prefixes`` is set and an argument is a
            configuration.
    """
    kwargs = {k: v for k, v in config.items() if k not in _RESERVED}
    init = config.get(OBJECT_INIT, "__init__")
    if allowed_prefixes is not None:
        _refuse_nested_configurations((), kwargs)
    target = import_object(config[OBJECT_TARGET], allowed_prefixes=allowed_prefixes)
    if cls is not object and init == "__init__":
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
                "instance because it is not a class and has no return type hint that is a "
                "class: it was not called"
            )
            raise TypeError(msg)
    return instantiate_object(target, _init_=init, **kwargs)


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

    A ``dict`` with the key ``"_target_"`` is always a factory
    configuration: it is built, and the result is checked against
    ``cls``. Any other object, including a ``dict`` without
    ``"_target_"``, is returned as-is if it is an instance of ``cls``.
    The key ``"_target_"`` is reserved: a ``dict`` that has it (for
    example ``Counter({"_target_": 2})``) cannot be passed through as
    data.

    Note:
        ``"_target_"`` is resolved at the top level only: a nested
        dictionary with a ``"_target_"`` key is passed to the target
        unchanged, it is not instantiated. With ``allowed_prefixes``
        it is refused with a ``ValueError``. Resolve the nested
        configurations with ``resolve_object`` (with the same
        ``allowed_prefixes``) and pass the results.

    Args:
        obj: Either a fully configured instance of ``cls``, or a
            ``dict`` containing a factory specification (with a
            ``"_target_"`` key pointing to the fully-qualified class
            name). The other keys are passed to the target by keyword,
            except ``"_init_"`` and ``"_allowed_prefixes_"``.
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
        TypeError: If the ``"_target_"`` is a class (or a function
            whose return type hint is a class) that is not a subclass
            of ``cls`` -- it is then not called --, or if the resolved
            object (including a ``dict`` without ``"_target_"``) is not
            an instance of ``cls``.
        ImportError: If the ``"_target_"`` does not match
            ``allowed_prefixes``, or if it cannot be found.
        ValueError: If ``allowed_prefixes`` is set and an argument of
            the configuration is itself a configuration (a ``dict``
            with a ``"_target_"`` key).

    Security:
        When ``obj`` is a ``dict`` with a ``"_target_"`` key, its
        value is imported and the target is called: this imports and
        instantiates arbitrary objects from a dotted path string with
        no allowlist by default. Only call this function with a
        configuration dictionary that comes from a trusted source;
        never resolve a ``_target_`` derived from untrusted input, as
        this is a remote-code-execution vector. If the configuration
        cannot be fully trusted, pass ``allowed_prefixes`` to restrict
        the ``"_target_"`` to the modules you expect, and check the
        arguments of the allowed objects too: an allow-list limits
        *what* can be called, not the arguments it is called with, and
        a prefix that contains a function that resolves its arguments
        (any ``resolve_*``, ``import_object``, ``pickle``, ``eval``,
        ...) makes it worthless: never allow ``"coola"`` or
        ``"coola.factory"``.

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
    cls_name = " | ".join(getattr(c, "__qualname__", repr(c)) for c in classes)
    # The key `_target_` is reserved: a dict that has it is always a configuration.
    if isinstance(obj, dict) and OBJECT_TARGET in obj:
        logger.info("Initializing a %s instance from its configuration...", cls_name)
        obj = _build(obj, cls, cls_name, allowed_prefixes=allowed_prefixes, strict=strict)
    if not isinstance(obj, cls):
        msg = f"Received object is not a {cls_name} instance (received: {type(cast('object', obj))}"
        if isinstance(obj, dict):
            msg += f"; a configuration needs the `{OBJECT_TARGET}` key"
        raise TypeError(msg + ")")
    return obj

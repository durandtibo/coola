r"""Contain helpers to deprecate the way a function is called."""

from __future__ import annotations

__all__ = ["deprecate_positional_options"]

import functools
import inspect
import warnings
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def deprecate_positional_options(*names: str) -> Callable[[F], F]:
    r"""Deprecate passing some options of a function by position.

    The options must be keyword-only in the signature of the decorated
    function (they follow a bare ``*``). The decorated function still
    accepts them by position, in the order of ``names``, and emits a
    ``DeprecationWarning``, so the callers have a release to move to the
    keywords. The warning is emitted when the function is called, also
    for a generator function.

    Args:
        *names: The names of the options that were positional, in their
            former order.

    Returns:
        The decorator.

    Example:
        ```pycon
        >>> import warnings
        >>> from coola.tools.deprecation import deprecate_positional_options
        >>> @deprecate_positional_options("scale")
        ... def multiply(value, *, scale=2):
        ...     return value * scale
        ...
        >>> multiply(3, scale=4)
        12
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     multiply(3, 4)
        ...
        12
        >>> caught[0].category.__name__
        'DeprecationWarning'

        ```
    """

    def decorator(func: F) -> F:
        num_positional = sum(
            parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for parameter in inspect.signature(func).parameters.values()
        )

        name = getattr(func, "__name__", repr(func))

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if len(args) > num_positional:
                args, kwargs = _move_to_keywords(
                    name, args, kwargs, num_positional=num_positional, names=names
                )
            return func(*args, **kwargs)

        return cast("F", wrapper)

    return decorator


def _move_to_keywords(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    num_positional: int,
    names: tuple[str, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    r"""Move the deprecated positional options to the keyword arguments.

    Raises:
        TypeError: if there are too many positional arguments, or if an
            option is given twice (by position and by keyword).
    """
    extra = args[num_positional:]
    if len(extra) > len(names):
        msg = (
            f"{name}() takes at most {num_positional + len(names)} positional arguments "
            f"but {len(args)} were given"
        )
        raise TypeError(msg)
    used = names[: len(extra)]
    warnings.warn(
        f"Passing {', '.join(f'`{n}`' for n in used)} to `{name}` as positional arguments is "
        "deprecated, use keyword arguments",
        DeprecationWarning,
        stacklevel=3,
    )
    kwargs = dict(kwargs)
    for option, value in zip(used, extra, strict=True):
        if option in kwargs:
            msg = f"{name}() got multiple values for argument {option!r}"
            raise TypeError(msg)
        kwargs[option] = value
    return args[:num_positional], kwargs

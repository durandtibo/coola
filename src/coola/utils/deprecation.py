r"""Contain utilities to deprecate public names."""

from __future__ import annotations

__all__ = ["deprecated_alias"]

import warnings
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")


def deprecated_alias(func: Callable[P, R], name: str) -> Callable[P, R]:
    r"""Create a deprecated alias of a function.

    The alias behaves exactly like ``func`` but emits a
    ``DeprecationWarning`` each time it is called.

    Args:
        func: The function to alias.
        name: The name of the alias.

    Returns:
        The deprecated alias, named ``name``.

    Example:
        ```pycon
        >>> import warnings
        >>> from coola.utils.deprecation import deprecated_alias
        >>> def get_value() -> int:
        ...     return 42
        ...
        >>> get_old_value = deprecated_alias(get_value, "get_old_value")
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     get_old_value()
        ...
        42
        >>> caught[0].category.__name__
        'DeprecationWarning'

        ```
    """
    func_name = getattr(func, "__name__", repr(func))
    module = getattr(func, "__module__", "").removesuffix(".interface")
    message = (
        f"`{name}` is deprecated and will be removed in a future release. "
        f"Use `{module}.{func_name}` instead."
    )

    @wraps(func)
    def alias(*args: P.args, **kwargs: P.kwargs) -> R:
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    alias.__name__ = name
    alias.__qualname__ = name
    alias.__doc__ = f"Deprecated alias of ``{func_name}``. {message}"
    return alias

r"""Contain utilities to manage optional dependencies."""

from __future__ import annotations

__all__ = ["check_package", "decorator_package_available", "module_available", "package_available"]

import importlib
from contextlib import suppress
from functools import lru_cache, wraps
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


@lru_cache
def package_available(name: str) -> bool:
    r"""Indicate if a package is available or not.

    Args:
        name: The package name to check.

    Returns:
        ``True`` if the package is available, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import package_available
        >>> package_available("os")
        True
        >>> package_available("missing_package")
        False

        ```
    """
    with suppress(ModuleNotFoundError):
        return find_spec(name) is not None
    return False


@lru_cache
def module_available(name: str) -> bool:
    r"""Indicate if a module is available or not.

    Args:
        name: The module name to check.

    Returns:
        ``True`` if the module is available, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import module_available
        >>> module_available("os")
        True
        >>> module_available("os.missing")
        False
        >>> module_available("missing.module")
        False

        ```
    """
    with suppress(ImportError):
        importlib.import_module(name)
        return True
    return False


def check_package(package: str, command: str | None = None) -> None:
    r"""Check if the given package is installed.

    Args:
        package: The package name.
        command: The command to install the package.

    Raises:
        RuntimeError: if the package is not installed.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_package
        >>> check_package("numpy")

        ```
    """
    if not package_available(package):
        msg = f"'{package}' package is required but not installed."
        if command is not None:
            msg += f" You can install '{package}' package with the command:\n\n{command}"
        raise RuntimeError(msg)


def decorator_package_available(fn: F, condition: Callable[[], bool]) -> F:
    r"""Implement a decorator to execute a function only if a package is
    installed.

    Args:
        fn: The function to execute.
        condition: The condition to
            check if a package is installed or not.

    Returns:
        A wrapper around ``fn`` if condition is true,
            otherwise ``None``.

    Example:
        ```pycon
        >>> from functools import partial
        >>> from coola.utils.imports import decorator_package_available, is_numpy_available
        >>> decorator = partial(decorator_package_available, condition=is_numpy_available)
        >>> @decorator
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function(2)
            44

        ```
    """

    @wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
        if not condition():
            return None
        return fn(*args, **kwargs)

    return inner


def raise_package_missing_error(package_name: str, install_cmd: str) -> NoReturn:
    r"""Raise a RuntimeError for a missing package.

    Args:
        package_name: The name of the missing package.
        install_cmd: The pip install command for the package.

    Raises:
        RuntimeError: Always raised to indicate the package is missing.
    """
    msg = (
        f"'{package_name}' package is required but not installed. "
        f"You can install '{package_name}' package with the command:\n\n"
        f"pip install {install_cmd}\n\nor\n\n"
        f"uv pip install {install_cmd}\n"
    )
    raise RuntimeError(msg)

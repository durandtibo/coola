r"""Contain utilities to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "MissingPackageError",
    "availability_cache",
    "check_package",
    "clear_availability_caches",
    "module_available",
    "package_available",
    "raise_package_missing_error",
]

import importlib
import logging
from contextlib import suppress
from functools import cache
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Never, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

logger: logging.Logger = logging.getLogger(__name__)

F = TypeVar("F", bound="Callable[..., Any]")


class _CachedFunction(Protocol):
    def cache_clear(self) -> None: ...


_CACHED_FUNCTIONS: list[_CachedFunction] = []


def availability_cache(fn: F) -> F:
    r"""Cache the result of an availability check and register it, so
    ``clear_availability_caches`` can reset it.

    Use it instead of ``functools.cache`` on the functions that check
    if a package is available.

    Args:
        fn: The availability check to cache.

    Returns:
        The cached function, which has a ``cache_clear`` method.

    Example:
        ```pycon
        >>> from coola.utils.imports import availability_cache
        >>> @availability_cache
        ... def is_my_package_available() -> bool:
        ...     return True
        ...
        >>> is_my_package_available()
        True

        ```
    """
    cached = cache(fn)
    _CACHED_FUNCTIONS.append(cast("_CachedFunction", cached))
    return cast("F", cached)


def clear_availability_caches() -> None:
    r"""Clear the cached results of the availability checks.

    ``package_available``, ``module_available`` and the
    ``is_<package>_available`` functions cache their result, so the
    availability of a package cannot change during the life of the
    process. Call this function to force the checks to run again, for
    example in a test that changes ``sys.modules`` or ``sys.path``.

    The functions to clear are the ones decorated with
    ``availability_cache``, so there is no list to maintain.

    Example:
        ```pycon
        >>> from coola.utils.imports import clear_availability_caches
        >>> clear_availability_caches()

        ```
    """
    for function in _CACHED_FUNCTIONS:
        function.cache_clear()


class MissingPackageError(ModuleNotFoundError, RuntimeError):
    r"""Raised when an optional package is required but not installed.

    It is a ``ModuleNotFoundError`` (so ``except ImportError`` works as
    expected). It also derives from ``RuntimeError``, which this error
    used to be, so existing ``except RuntimeError`` code keeps working.

    Example:
        ```pycon
        >>> from coola.utils.imports import MissingPackageError, check_package
        >>> try:
        ...     check_package("not_an_installed_package")
        ... except ImportError as error:
        ...     print(type(error).__name__, error.name)
        ...
        MissingPackageError not_an_installed_package

        ```
    """


@availability_cache
def package_available(name: str) -> bool:
    r"""Indicate if a package is installed or not.

    The package is only looked up (it is not imported), so a package that
    is installed but broken is reported as available. Use
    ``module_available`` to check that a module can be imported.

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
    # ``find_spec`` raises ``ValueError`` if the module is in ``sys.modules`` with
    # no ``__spec__``, and ``ImportError`` if a parent package cannot be imported.
    with suppress(ImportError, ValueError):
        return find_spec(name) is not None
    return False


@availability_cache
def module_available(name: str) -> bool:
    r"""Indicate if a module is available or not.

    The module is imported to check that it works, so a package that is
    installed but broken is reported as not available. The result is
    cached; use ``clear_availability_caches`` to reset it.

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
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    except Exception:
        # A broken installation (e.g. a missing shared library or an ABI
        # mismatch) can fail with other exceptions: treat the module as
        # unavailable, but tell why.
        logger.warning("The module %r is installed but cannot be imported", name, exc_info=True)
        return False
    return True


def check_package(package: str, *, install: str | None = None) -> None:
    r"""Check if the given package is installed and can be imported.

    Args:
        package: The package name.
        install: The name to give to ``pip install`` when it differs
            from ``package``. By default, it is ``package``.

    Raises:
        MissingPackageError: if the package is not available.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_package
        >>> check_package("os")

        ```
    """
    if not module_available(package):
        raise_package_missing_error(package, install=install)


def raise_package_missing_error(package: str, *, install: str | None = None) -> Never:
    r"""Raise a ``MissingPackageError`` for a missing package.

    Args:
        package: The name of the missing package.
        install: The name to give to ``pip install`` when it differs
            from ``package``. By default, it is ``package``.

    Raises:
        MissingPackageError: Always raised to indicate the package is
            missing.

    Example:
        ```pycon
        >>> from coola.utils.imports import raise_package_missing_error
        >>> raise_package_missing_error("numpy")  # doctest: +SKIP

        ```
    """
    install = package if install is None else install
    msg = (
        f"'{package}' package is required but not installed. "
        f"You can install '{package}' package with the command:\n\n"
        f"pip install {install}\n\nor\n\n"
        f"uv pip install {install}\n"
    )
    raise MissingPackageError(msg, name=package)

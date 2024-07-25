r"""Contain functions to manage package versions."""

from __future__ import annotations

__all__ = ["compare_version", "get_package_version"]


from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING
from unittest.mock import Mock

from coola.utils.imports import check_packaging, is_packaging_available

if TYPE_CHECKING:
    from collections.abc import Callable

if is_packaging_available():
    from packaging.version import Version
else:  # pragma: no cover
    Version = Mock()


def compare_version(package: str, op: Callable, version: str) -> bool:
    r"""Compare a package version to a given version.

    Args:
        package: Specifies the package to check.
        op: Specifies the comparison operator.
        version: Specifies the version to compare with.

    Returns:
        The comparison status.

    Example usage:

    ```pycon

    >>> import operator
    >>> from coola.utils.version import compare_version
    >>> compare_version("pytest", op=operator.ge, version="7.3.0")
    True

    ```
    """
    check_packaging()
    pkg_version = get_package_version(package)
    if pkg_version is None:
        return False
    return op(pkg_version, Version(version))


def get_package_version(package: str) -> Version | None:
    r"""Get the package version.

    Args:
        package: Specifies the package name.

    Returns:
        The package version.

    Example usage:

    ```pycon

    >>> from coola.utils.version import get_package_version
    >>> get_package_version("pytest")
    <Version('...')>

    ```
    """
    check_packaging()
    try:
        return Version(version(package))
    except PackageNotFoundError:
        return None

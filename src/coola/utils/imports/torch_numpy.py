r"""Contain utilities for optional torch dependency."""

from __future__ import annotations

__all__ = ["check_torch_numpy", "is_torch_numpy_available", "torch_numpy_available"]

from contextlib import suppress
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar

from coola.utils.imports.numpy import is_numpy_available
from coola.utils.imports.torch import is_torch_available
from coola.utils.imports.universal import decorator_package_available

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


if is_torch_available():
    import torch
else:  # pragma: no cover
    from coola.utils.fallback.torch import torch


def check_torch_numpy() -> None:
    r"""Check if the ``torch`` and ``numpy`` packages are installed and
    are compatible.

    Raises:
        RuntimeError: if one of the packages is not installed or if
            they are not compatible.

    Example:
        ```pycon
        >>> from coola.utils.imports import check_torch_numpy
        >>> check_torch_numpy()

        ```
    """
    if not is_torch_numpy_available():
        msg = (
            "'torch' and 'numpy' packages are required but one of the package is not "
            "installed or they are not compatible. "
        )
        raise RuntimeError(msg)


@lru_cache
def is_torch_numpy_available() -> bool:
    r"""Indicate if the ``torch`` and ``numpy`` packages are installed
    and are compatible.

    Returns:
        ``True`` if both packages are available and compatible,
            otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.utils.imports import is_torch_numpy_available
        >>> is_torch_numpy_available()

        ```
    """
    if not is_torch_available():
        return False
    if not is_numpy_available():
        return False
    with suppress(RuntimeError):
        torch.tensor([1.0]).numpy()  # Check if the libraries are compatible
        return True
    return False


def torch_numpy_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``torch`` and
    ``numpy`` packages are installed and are compatible.

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``torch`` and ``numpy`` packages
            are installed and are compatible., otherwise ``None``.

    Example:
        ```pycon
        >>> from coola.utils.imports import torch_numpy_available
        >>> @torch_numpy_available
        ... def my_function(n: int = 0) -> int:
        ...     return 42 + n
        ...
        >>> my_function()

        ```
    """
    return decorator_package_available(fn, is_torch_numpy_available)

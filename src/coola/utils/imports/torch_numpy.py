r"""Contain utilities to check that the optional ``torch`` and ``numpy``
dependencies are installed and mutually compatible.

Unlike the other modules of ``coola.utils.imports``, which check one
package, these utilities check a **pair** of packages: ``torch`` and
``numpy`` are both installed, *and* ``torch`` can exchange data with
``numpy`` (a ``torch`` built against another major version of
``numpy`` cannot). Use them for code that converts between tensors and
arrays. If the code only needs one of the two, use the check of this
package instead (``is_torch_available`` or ``is_numpy_available``).
"""

from __future__ import annotations

__all__ = ["check_torch_numpy", "is_torch_numpy_available", "torch_numpy_available"]

from contextlib import suppress
from typing import TYPE_CHECKING, Any, TypeVar

from coola.utils.imports.numpy import is_numpy_available
from coola.utils.imports.torch import is_torch_available
from coola.utils.imports.universal import (
    MissingPackageError,
    availability_cache,
    decorator_package_available,
)

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


def check_torch_numpy() -> None:
    r"""Check if the ``torch`` and ``numpy`` packages are installed and
    are compatible.

    Unlike the other ``check_<package>`` functions, this function
    checks a pair of packages (see ``is_torch_numpy_available``).

    Raises:
        MissingPackageError: if one of the packages is not installed or if
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
        raise MissingPackageError(msg)


@availability_cache
def is_torch_numpy_available() -> bool:
    r"""Indicate if the ``torch`` and ``numpy`` packages are installed
    and are compatible.

    The packages are compatible if a ``torch`` tensor can be converted
    to a ``numpy`` array. Unlike the other ``is_<package>_available``
    functions, this function checks a pair of packages.

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
    # Imported here and not at the top of the module, so importing
    # ``coola.utils.imports`` does not import ``torch``.
    import torch  # noqa: PLC0415

    with suppress(RuntimeError):
        torch.tensor([1.0]).numpy()  # Check if the libraries are compatible
        return True
    return False


def torch_numpy_available(fn: F) -> F:
    r"""Implement a decorator to execute a function only if ``torch`` and
    ``numpy`` packages are installed and are compatible.

    Unlike the other ``<package>_available`` decorators, this decorator
    checks a pair of packages (see ``is_torch_numpy_available``).

    Args:
        fn: The function to execute.

    Returns:
        A wrapper around ``fn`` if ``torch`` and ``numpy`` packages
            are installed and are compatible, otherwise ``None``.

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

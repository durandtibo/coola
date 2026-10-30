r"""Define some pytest fixtures for testing.

`pytest` is required to use these fixtures.

Each optional dependency has a pair of marks: ``<name>_available`` skips
the test unless the dependency is available, and
``<name>_not_available`` skips it if the dependency is available. They
are created by ``requires`` from one row of ``_DEPENDENCIES``, so
supporting a new dependency takes one line.

Most marks are named after one package. The ``torch_cuda_*``,
``torch_mps_*`` and ``torch_numpy_*`` marks are named after a feature of
PyTorch or a pair of packages: they check that PyTorch is installed
*and* that the CUDA device, the MPS device or NumPy (in a compatible
version) is usable.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, NamedTuple

import pytest

from coola.utils.imports import (
    is_colorlog_available,
    is_jax_available,
    is_numpy_available,
    is_packaging_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_pydantic_available,
    is_rich_available,
    is_scipy_available,
    is_torch_available,
    is_torch_numpy_available,
    is_xarray_available,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def requires(is_available: bool, name: str) -> tuple[pytest.MarkDecorator, pytest.MarkDecorator]:
    r"""Create the pair of marks for a dependency.

    Use it to support a new optional dependency: the marks are meant to
    be assigned to module-level names, so they can be imported and used
    as test decorators.

    Args:
        is_available: Whether the dependency is available.
        name: The name of the dependency in the skip reasons, e.g.
            ``"PyTorch with CUDA"``.

    Returns:
        The ``(available, not_available)`` marks. The first one skips
            the test if the dependency is not available, the second one
            skips it if the dependency is available.

    Example:
        ```pycon
        >>> from coola.testing.marks import requires
        >>> numpy_available, numpy_not_available = requires(True, "NumPy")
        >>> numpy_available.kwargs["reason"]
        'Requires NumPy'
        >>> numpy_not_available.kwargs["reason"]
        'Skip if NumPy is available'

        ```
    """
    return (
        pytest.mark.skipif(not is_available, reason=f"Requires {name}"),
        pytest.mark.skipif(is_available, reason=f"Skip if {name} is available"),
    )


class _Dependency(NamedTuple):
    r"""Describe an optional dependency used by a mark.

    Attributes:
        is_available: The check of the library, called on first use.
        name: The singular name of the dependency, used in the reasons.
    """

    is_available: Callable[[], bool]  # the check of the library, called on first use
    name: str  # singular, used in the reasons


def _is_cuda_available() -> bool:
    r"""Indicate if CUDA is available.

    ``coola.utils.tensor`` is imported on the first call, because it
    imports ``torch`` when it is installed.

    Returns:
        ``True`` if CUDA is available, otherwise ``False``.
    """
    # ``coola.utils.tensor`` imports ``torch`` when it is installed: import it on first call.
    return import_module("coola.utils.tensor").is_cuda_available()


def _is_mps_available() -> bool:
    r"""Indicate if MPS is available.

    Returns:
        ``True`` if MPS is available, otherwise ``False``.
    """
    return import_module("coola.utils.tensor").is_mps_available()


# The availability of a package is checked when its marks are first accessed, not when this
# module is imported: the checks import the packages (``torch``, ``jax``...) and
# ``is_mps_available`` allocates a tensor on the device, which is slow and initializes a GPU
# context. A test module that only needs ``numpy_available`` does not pay for them.
# One row per dependency: the only place to edit to support a new one.
_DEPENDENCIES: dict[str, _Dependency] = {
    "colorlog": _Dependency(is_colorlog_available, "colorlog"),
    "jax": _Dependency(is_jax_available, "JAX"),
    "numpy": _Dependency(is_numpy_available, "NumPy"),
    "packaging": _Dependency(is_packaging_available, "packaging"),
    "pandas": _Dependency(is_pandas_available, "pandas"),
    "polars": _Dependency(is_polars_available, "polars"),
    "pyarrow": _Dependency(is_pyarrow_available, "pyarrow"),
    "pydantic": _Dependency(is_pydantic_available, "pydantic"),
    "rich": _Dependency(is_rich_available, "rich"),
    "scipy": _Dependency(is_scipy_available, "SciPy"),
    "torch": _Dependency(is_torch_available, "PyTorch"),
    "torch_cuda": _Dependency(_is_cuda_available, "PyTorch with CUDA"),
    "torch_mps": _Dependency(_is_mps_available, "PyTorch with MPS"),
    "torch_numpy": _Dependency(is_torch_numpy_available, "PyTorch with NumPy"),
    "xarray": _Dependency(is_xarray_available, "xarray"),
}


__all__ = ["requires"]
__all__ += [
    f"{key}{suffix}" for key in _DEPENDENCIES for suffix in ("_available", "_not_available")
]


def __getattr__(name: str) -> pytest.MarkDecorator:
    # The marks are still module attributes (``from coola.testing.marks import
    # torch_available`` works, and ``pytestmark = [torch_available]``), through PEP 562.
    for suffix in ("_available", "_not_available"):
        key = name.removesuffix(suffix)
        if name.endswith(suffix) and key in _DEPENDENCIES:
            dependency = _DEPENDENCIES[key]
            available, not_available = requires(dependency.is_available(), dependency.name)
            # Cached: ``__getattr__`` is not called again for these names.
            globals()[f"{key}_available"] = available
            globals()[f"{key}_not_available"] = not_available
            return globals()[name]
    msg = (
        f"module {__name__!r} has no attribute {name!r}, the valid names are: {', '.join(__all__)}"
    )
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(__all__)

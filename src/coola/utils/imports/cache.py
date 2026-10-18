r"""Contain a function to reset the cached availability of the optional
dependencies."""

from __future__ import annotations

__all__ = ["clear_availability_caches"]

from coola.utils.imports.colorlog import is_colorlog_available
from coola.utils.imports.jax import is_jax_available
from coola.utils.imports.numpy import is_numpy_available
from coola.utils.imports.packaging import is_packaging_available
from coola.utils.imports.pandas import is_pandas_available
from coola.utils.imports.polars import is_polars_available
from coola.utils.imports.pyarrow import is_pyarrow_available
from coola.utils.imports.pydantic import is_pydantic_available
from coola.utils.imports.rich import is_rich_available
from coola.utils.imports.torch import is_torch_available
from coola.utils.imports.torch_numpy import is_torch_numpy_available
from coola.utils.imports.universal import module_available, package_available
from coola.utils.imports.xarray import is_xarray_available

_CACHED_FUNCTIONS = (
    is_colorlog_available,
    is_jax_available,
    is_numpy_available,
    is_packaging_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_pydantic_available,
    is_rich_available,
    is_torch_available,
    is_torch_numpy_available,
    is_xarray_available,
    module_available,
    package_available,
)


def clear_availability_caches() -> None:
    r"""Clear the cached results of the availability checks.

    ``package_available``, ``module_available`` and the
    ``is_<package>_available`` functions cache their result, so the
    availability of a package cannot change during the life of the
    process. Call this function to force the checks to run again, for
    example in a test that changes ``sys.modules`` or ``sys.path``.

    Example:
        ```pycon
        >>> from coola.utils.imports import clear_availability_caches
        >>> clear_availability_caches()

        ```
    """
    for function in _CACHED_FUNCTIONS:
        function.cache_clear()

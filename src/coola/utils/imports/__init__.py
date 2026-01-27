r"""Implement some utility functions to manage optional dependencies."""

from __future__ import annotations

__all__ = [
    "LazyModule",
    "check_jax",
    "check_numpy",
    "check_package",
    "check_packaging",
    "check_pandas",
    "check_polars",
    "check_pyarrow",
    "check_requests",
    "check_torch",
    "check_torch_numpy",
    "check_urllib3",
    "check_xarray",
    "decorator_package_available",
    "is_jax_available",
    "is_numpy_available",
    "is_packaging_available",
    "is_pandas_available",
    "is_polars_available",
    "is_pyarrow_available",
    "is_requests_available",
    "is_torch_available",
    "is_torch_numpy_available",
    "is_urllib3_available",
    "is_xarray_available",
    "jax_available",
    "lazy_import",
    "module_available",
    "numpy_available",
    "package_available",
    "packaging_available",
    "pandas_available",
    "polars_available",
    "pyarrow_available",
    "raise_jax_missing_error",
    "raise_numpy_missing_error",
    "raise_package_missing_error",
    "raise_packaging_missing_error",
    "raise_pandas_missing_error",
    "raise_polars_missing_error",
    "raise_pyarrow_missing_error",
    "raise_requests_missing_error",
    "raise_torch_missing_error",
    "raise_urllib3_missing_error",
    "raise_xarray_missing_error",
    "requests_available",
    "torch_available",
    "torch_numpy_available",
    "urllib3_available",
    "xarray_available",
]

from coola.utils.imports.jax import (
    check_jax,
    is_jax_available,
    jax_available,
    raise_jax_missing_error,
)
from coola.utils.imports.lazy import LazyModule, lazy_import
from coola.utils.imports.numpy import (
    check_numpy,
    is_numpy_available,
    numpy_available,
    raise_numpy_missing_error,
)
from coola.utils.imports.packaging import (
    check_packaging,
    is_packaging_available,
    packaging_available,
    raise_packaging_missing_error,
)
from coola.utils.imports.pandas import (
    check_pandas,
    is_pandas_available,
    pandas_available,
    raise_pandas_missing_error,
)
from coola.utils.imports.polars import (
    check_polars,
    is_polars_available,
    polars_available,
    raise_polars_missing_error,
)
from coola.utils.imports.pyarrow import (
    check_pyarrow,
    is_pyarrow_available,
    pyarrow_available,
    raise_pyarrow_missing_error,
)
from coola.utils.imports.requests import (
    check_requests,
    is_requests_available,
    raise_requests_missing_error,
    requests_available,
)
from coola.utils.imports.torch import (
    check_torch,
    is_torch_available,
    raise_torch_missing_error,
    torch_available,
)
from coola.utils.imports.torch_numpy import (
    check_torch_numpy,
    is_torch_numpy_available,
    torch_numpy_available,
)
from coola.utils.imports.universal import (
    check_package,
    decorator_package_available,
    module_available,
    package_available,
    raise_package_missing_error,
)
from coola.utils.imports.urllib3 import (
    check_urllib3,
    is_urllib3_available,
    raise_urllib3_missing_error,
    urllib3_available,
)
from coola.utils.imports.xarray import (
    check_xarray,
    is_xarray_available,
    raise_xarray_missing_error,
    xarray_available,
)

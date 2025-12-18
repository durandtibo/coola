r"""Define some utility functions for testing."""

from __future__ import annotations

__all__ = [
    "jax_available",
    "jax_not_available",
    "numpy_available",
    "numpy_not_available",
    "packaging_available",
    "packaging_not_available",
    "pandas_available",
    "pandas_not_available",
    "polars_available",
    "polars_not_available",
    "pyarrow_available",
    "pyarrow_not_available",
    "torch_available",
    "torch_cuda_available",
    "torch_mps_available",
    "torch_not_available",
    "torch_numpy_available",
    "xarray_available",
    "xarray_not_available",
]

import pytest

from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_packaging_available,
    is_pandas_available,
    is_polars_available,
    is_pyarrow_available,
    is_torch_available,
    is_torch_numpy_available,
    is_xarray_available,
)
from coola.utils.tensor import is_cuda_available, is_mps_available

jax_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_jax_available(), reason="Requires JAX"
)
jax_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_jax_available(), reason="Skip if JAX is available"
)

numpy_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_numpy_available(), reason="Requires NumPy"
)
numpy_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_numpy_available(), reason="Skip if NumPy is available"
)

packaging_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_packaging_available(), reason="Requires packaging"
)
packaging_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_packaging_available(), reason="Skip if packaging is available"
)

pandas_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_pandas_available(), reason="Requires pandas"
)
pandas_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_pandas_available(), reason="Skip if pandas is available"
)

polars_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_polars_available(), reason="Requires polars"
)
polars_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_polars_available(), reason="Skip if polars is available"
)

pyarrow_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_pyarrow_available(), reason="Requires pyarrow"
)
pyarrow_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_pyarrow_available(), reason="Skip if pyarrow is available"
)

torch_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_torch_available(), reason="Requires PyTorch"
)
torch_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_torch_available(), reason="Skip if PyTorch is available"
)
torch_cuda_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_cuda_available(), reason="Requires PyTorch and CUDA"
)
torch_numpy_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_torch_numpy_available(), reason="Requires PyTorch and NumPy"
)
torch_mps_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_mps_available(), reason="Requires PyTorch and MPS"
)

xarray_available: pytest.MarkDecorator = pytest.mark.skipif(
    not is_xarray_available(), reason="Requires xarray"
)
xarray_not_available: pytest.MarkDecorator = pytest.mark.skipif(
    is_xarray_available(), reason="Skip if xarray is available"
)

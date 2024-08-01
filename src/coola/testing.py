r"""Define some utility functions for testing."""

from __future__ import annotations

__all__ = [
    "jax_available",
    "numpy_available",
    "packaging_available",
    "pandas_available",
    "polars_available",
    "polars_greater_equal_0_20_0",
    "pyarrow_available",
    "torch_available",
    "torch_cuda_available",
    "torch_mps_available",
    "torch_numpy_available",
    "xarray_available",
]

import pytest

from coola.equality.handlers.polars_ import POLARS_GREATER_EQUAL_0_20_0
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

jax_available = pytest.mark.skipif(not is_jax_available(), reason="Requires JAX")
numpy_available = pytest.mark.skipif(not is_numpy_available(), reason="Requires NumPy")
packaging_available = pytest.mark.skipif(not is_packaging_available(), reason="Requires packaging")
pandas_available = pytest.mark.skipif(not is_pandas_available(), reason="Requires pandas")
polars_available = pytest.mark.skipif(not is_polars_available(), reason="Requires polars")
torch_available = pytest.mark.skipif(not is_torch_available(), reason="Requires PyTorch")
torch_cuda_available = pytest.mark.skipif(
    not is_cuda_available(), reason="Requires PyTorch and CUDA"
)
torch_numpy_available = pytest.mark.skipif(
    not is_torch_numpy_available(), reason="Requires PyTorch and NumPy"
)
torch_mps_available = pytest.mark.skipif(not is_mps_available(), reason="Requires PyTorch and MPS")
xarray_available = pytest.mark.skipif(not is_xarray_available(), reason="Requires xarray")
pyarrow_available = pytest.mark.skipif(not is_pyarrow_available(), reason="Requires pyarrow")

polars_greater_equal_0_20_0 = pytest.mark.skipif(
    not POLARS_GREATER_EQUAL_0_20_0, reason="Requires polars>=0.20.0"
)

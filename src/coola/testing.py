from __future__ import annotations

__all__ = [
    "jax_available",
    "numpy_available",
    "pandas_available",
    "polars_available",
    "torch_available",
    "torch_cuda_available",
    "torch_mps_available",
    "xarray_available",
]

from pytest import mark

from coola.utils.imports import (
    is_jax_available,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available,
)
from coola.utils.tensor import is_cuda_available, is_mps_available

jax_available = mark.skipif(not is_jax_available(), reason="Requires JAX")
numpy_available = mark.skipif(not is_numpy_available(), reason="Requires NumPy")
pandas_available = mark.skipif(not is_pandas_available(), reason="Requires pandas")
polars_available = mark.skipif(not is_polars_available(), reason="Requires polars")
torch_available = mark.skipif(not is_torch_available(), reason="Requires PyTorch")
torch_cuda_available = mark.skipif(not is_cuda_available(), reason="Requires PyTorch and CUDA")
torch_mps_available = mark.skipif(not is_mps_available(), reason="Requires PyTorch and MPS")
xarray_available = mark.skipif(not is_xarray_available(), reason="Requires xarray")

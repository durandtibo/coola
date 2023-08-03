from __future__ import annotations

__all__ = [
    "jax_available",
    "numpy_available",
    "pandas_available",
    "polars_available",
    "torch_available",
    "xarray_available",
]

from pytest import mark

from coola.utils.imports import (
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available, is_jax_available,
)

jax_available = mark.skipif(not is_jax_available(), reason="Requires JAX")
numpy_available = mark.skipif(not is_numpy_available(), reason="Requires NumPy")
pandas_available = mark.skipif(not is_pandas_available(), reason="Requires pandas")
polars_available = mark.skipif(not is_polars_available(), reason="Requires polars")
torch_available = mark.skipif(not is_torch_available(), reason="Requires PyTorch")
xarray_available = mark.skipif(not is_xarray_available(), reason="Requires xarray")

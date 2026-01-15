# Testing Utilities

:book: This page describes the `coola.testing` package, which provides utilities to make testing
with `coola` easier, particularly when dealing with optional dependencies.

**Prerequisites:** You'll need to know a bit of Python and pytest.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/) and
[pytest documentation](https://docs.pytest.org/).

## Overview

The `coola.testing` package provides pytest fixtures and markers to help you write tests that work
with optional dependencies. It offers:

1. **Dependency markers** - Skip tests based on whether optional packages are available
2. **Consistent testing** - Standard markers for common dependencies
3. **Clean test code** - Avoid manual dependency checks in your tests

This is particularly useful when writing tests for libraries that support multiple backends or have
optional features that depend on specific packages.

## Available Fixtures

The package provides pytest markers that automatically skip tests based on package availability.
These markers come in pairs: one to require a package and one to skip when a package is available.

### NumPy Markers

#### `numpy_available`

Skip the test if NumPy is not available:

```python
from coola.testing import numpy_available


@numpy_available
def test_numpy_functionality():
    import numpy as np

    # This test only runs if NumPy is installed
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
```

#### `numpy_not_available`

Skip the test if NumPy is available (useful for testing fallback behavior):

```python
from coola.testing import numpy_not_available


@numpy_not_available
def test_fallback_without_numpy():
    # This test only runs if NumPy is NOT installed
    # Test fallback implementation that doesn't use NumPy
    pass
```

### PyTorch Markers

#### `torch_available`

Skip the test if PyTorch is not available:

```python
from coola.testing import torch_available


@torch_available
def test_torch_functionality():
    import torch

    # This test only runs if PyTorch is installed
    tensor = torch.tensor([1, 2, 3])
    assert tensor.sum().item() == 6
```

#### `torch_not_available`

Skip the test if PyTorch is available:

```python
from coola.testing import torch_not_available


@torch_not_available
def test_fallback_without_torch():
    # This test only runs if PyTorch is NOT installed
    pass
```

#### `torch_cuda_available`

Skip the test if PyTorch with CUDA is not available:

```python
from coola.testing import torch_cuda_available


@torch_cuda_available
def test_gpu_functionality():
    import torch

    # This test only runs if PyTorch is installed and CUDA is available
    device = torch.device("cuda")
    tensor = torch.tensor([1, 2, 3], device=device)
    assert tensor.is_cuda
```

#### `torch_mps_available`

Skip the test if PyTorch with MPS (Apple Silicon) is not available:

```python
from coola.testing import torch_mps_available


@torch_mps_available
def test_mps_functionality():
    import torch

    # This test only runs if PyTorch is installed and MPS is available
    device = torch.device("mps")
    tensor = torch.tensor([1, 2, 3], device=device)
    assert tensor.is_mps
```

#### `torch_numpy_available`

Skip the test if both PyTorch and NumPy are not available:

```python
from coola.testing import torch_numpy_available


@torch_numpy_available
def test_torch_numpy_interop():
    import torch
    import numpy as np

    # This test only runs if both PyTorch and NumPy are installed
    arr = np.array([1, 2, 3])
    tensor = torch.from_numpy(arr)
    assert tensor.tolist() == [1, 2, 3]
```

### pandas Markers

#### `pandas_available`

Skip the test if pandas is not available:

```python
from coola.testing import pandas_available


@pandas_available
def test_pandas_functionality():
    import pandas as pd

    # This test only runs if pandas is installed
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert len(df) == 3
```

#### `pandas_not_available`

Skip the test if pandas is available:

```python
from coola.testing import pandas_not_available


@pandas_not_available
def test_fallback_without_pandas():
    # This test only runs if pandas is NOT installed
    pass
```

### polars Markers

#### `polars_available`

Skip the test if polars is not available:

```python
from coola.testing import polars_available


@polars_available
def test_polars_functionality():
    import polars as pl

    # This test only runs if polars is installed
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert len(df) == 3
```

#### `polars_not_available`

Skip the test if polars is available:

```python
from coola.testing import polars_not_available


@polars_not_available
def test_fallback_without_polars():
    # This test only runs if polars is NOT installed
    pass
```

### JAX Markers

#### `jax_available`

Skip the test if JAX is not available:

```python
from coola.testing import jax_available


@jax_available
def test_jax_functionality():
    import jax.numpy as jnp

    # This test only runs if JAX is installed
    arr = jnp.array([1, 2, 3])
    assert arr.sum() == 6
```

#### `jax_not_available`

Skip the test if JAX is available:

```python
from coola.testing import jax_not_available


@jax_not_available
def test_fallback_without_jax():
    # This test only runs if JAX is NOT installed
    pass
```

### xarray Markers

#### `xarray_available`

Skip the test if xarray is not available:

```python
from coola.testing import xarray_available


@xarray_available
def test_xarray_functionality():
    import xarray as xr

    # This test only runs if xarray is installed
    data = xr.DataArray([1, 2, 3])
    assert data.sum().item() == 6
```

#### `xarray_not_available`

Skip the test if xarray is available:

```python
from coola.testing import xarray_not_available


@xarray_not_available
def test_fallback_without_xarray():
    # This test only runs if xarray is NOT installed
    pass
```

### PyArrow Markers

#### `pyarrow_available`

Skip the test if pyarrow is not available:

```python
from coola.testing import pyarrow_available


@pyarrow_available
def test_pyarrow_functionality():
    import pyarrow as pa

    # This test only runs if pyarrow is installed
    arr = pa.array([1, 2, 3])
    assert len(arr) == 3
```

#### `pyarrow_not_available`

Skip the test if pyarrow is available:

```python
from coola.testing import pyarrow_not_available


@pyarrow_not_available
def test_fallback_without_pyarrow():
    # This test only runs if pyarrow is NOT installed
    pass
```

### packaging Markers

#### `packaging_available`

Skip the test if packaging is not available:

```python
from coola.testing import packaging_available


@packaging_available
def test_packaging_functionality():
    from packaging import version

    # This test only runs if packaging is installed
    v = version.parse("1.0.0")
    assert v.major == 1
```

#### `packaging_not_available`

Skip the test if packaging is available:

```python
from coola.testing import packaging_not_available


@packaging_not_available
def test_fallback_without_packaging():
    # This test only runs if packaging is NOT installed
    pass
```

## Usage Examples

### Testing with Multiple Dependencies

You can use multiple markers on a single test:

```python
from coola.testing import torch_available, numpy_available


@torch_available
@numpy_available
def test_torch_numpy_conversion():
    import torch
    import numpy as np

    # This test only runs if both PyTorch and NumPy are installed
    arr = np.array([1, 2, 3])
    tensor = torch.from_numpy(arr)
    back_to_numpy = tensor.numpy()
    assert np.array_equal(arr, back_to_numpy)
```

### Testing Fallback Behavior

Test that your code works when optional dependencies are not available:

```python
from coola.testing import torch_available, torch_not_available


@torch_available
def test_with_torch_backend():
    # Test using PyTorch backend
    from mylib import process_data

    result = process_data([1, 2, 3], backend="torch")
    assert result is not None


@torch_not_available
def test_with_native_backend():
    # Test using native Python backend when PyTorch is not available
    from mylib import process_data

    result = process_data([1, 2, 3], backend="native")
    assert result is not None
```

### Testing Platform-Specific Features

Test GPU or accelerator-specific functionality:

```python
from coola.testing import torch_cuda_available, torch_mps_available


@torch_cuda_available
def test_cuda_acceleration():
    import torch

    # Test CUDA-specific functionality
    device = torch.device("cuda")
    data = torch.randn(100, 100, device=device)
    assert data.is_cuda


@torch_mps_available
def test_mps_acceleration():
    import torch

    # Test MPS (Apple Silicon) specific functionality
    device = torch.device("mps")
    data = torch.randn(100, 100, device=device)
    assert data.is_mps
```

## Common Use Cases

### Library with Optional Dependencies

When writing a library that supports multiple backends:

```python
# mylib/tests/test_equality.py
from coola.testing import torch_available, numpy_available


@torch_available
def test_equality_torch_tensors():
    import torch
    from mylib import are_equal

    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([1, 2, 3])
    assert are_equal(t1, t2)


@numpy_available
def test_equality_numpy_arrays():
    import numpy as np
    from mylib import are_equal

    a1 = np.array([1, 2, 3])
    a2 = np.array([1, 2, 3])
    assert are_equal(a1, a2)
```

### Testing Error Messages

Test that appropriate errors are raised when dependencies are missing:

```python
from coola.testing import torch_not_available
import pytest


@torch_not_available
def test_error_without_torch():
    from mylib import TorchProcessor

    # Should raise an ImportError or similar when PyTorch is not available
    with pytest.raises(ImportError):
        processor = TorchProcessor()
```

### Comprehensive Test Coverage

Ensure both implementation and fallback paths are tested:

```python
from coola.testing import pandas_available, pandas_not_available


@pandas_available
def test_load_data_with_pandas():
    from mylib import load_data

    # When pandas is available, should return a DataFrame
    data = load_data("data.csv")
    import pandas as pd

    assert isinstance(data, pd.DataFrame)


@pandas_not_available
def test_load_data_without_pandas():
    from mylib import load_data

    # When pandas is not available, should return a dict
    data = load_data("data.csv")
    assert isinstance(data, dict)
```

## Design Principles

The `coola.testing` package follows these design principles:

1. **Declarative**: Use decorators to express test requirements clearly
2. **Standard integration**: Works seamlessly with pytest
3. **Comprehensive coverage**: Markers for all major scientific Python packages
4. **Paired markers**: Both "available" and "not available" variants for complete testing
5. **Clear skip messages**: Informative skip messages explain why tests were skipped

## See Also

- [pytest documentation](https://docs.pytest.org/): Learn more about pytest
- [`coola.utils.imports`](utils.md): For programmatically checking package availability
- [`coola.equality`](equality.md): For equality testing with multiple backends

# Utility Functions

:book: This page describes the `coola.utils` package, which provides various utility functions for
working with imports, formatting, arrays, and more.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.utils` package provides a collection of utility functions organized into several
submodules:

1. **Import utilities** - Check if packages are available and handle optional dependencies
2. **Formatting utilities** - Format objects, sequences, and mappings for display
3. **Array utilities** - Utilities for working with arrays
4. **Other utilities** - Path, version, environment variable, and introspection utilities

## Import Utilities (`coola.utils.imports`)

### Checking Package Availability

Use `package_available()` and `module_available()` to check if packages or modules are available:

```pycon

>>> from coola.utils.imports import package_available, module_available
>>> package_available("os")
True
>>> package_available("missing_package")
False
>>> module_available("os.path")
True

```

These functions use caching (`@lru_cache`) to avoid repeated checks, making them efficient for
repeated calls.

### Specific Package Checks

For common scientific packages, use dedicated check functions:

```pycon

>>> from coola.utils.imports import (
...     is_numpy_available,
...     is_torch_available,
...     is_pandas_available,
... )
>>> is_numpy_available()  # doctest: +SKIP
True
>>> is_torch_available()  # doctest: +SKIP
False

```

Available check functions:
- `is_numpy_available()` - Check if NumPy is available
- `is_torch_available()` - Check if PyTorch is available
- `is_pandas_available()` - Check if pandas is available
- `is_polars_available()` - Check if polars is available
- `is_jax_available()` - Check if JAX is available
- `is_xarray_available()` - Check if xarray is available
- `is_pyarrow_available()` - Check if PyArrow is available
- `is_packaging_available()` - Check if packaging is available
- `is_torch_numpy_available()` - Check if both PyTorch and NumPy are available

### Context Managers for Availability

Use context managers to conditionally execute code based on package availability:

```pycon

>>> from coola.utils.imports import numpy_available
>>> with numpy_available():  # doctest: +SKIP
...     import numpy as np
...     arr = np.array([1, 2, 3])
...     print(arr)
...

```

If the package is not available, the code block is skipped without raising an error.

Available context managers:
- `numpy_available()` - Execute block if NumPy is available
- `torch_available()` - Execute block if PyTorch is available
- `pandas_available()` - Execute block if pandas is available
- `polars_available()` - Execute block if polars is available
- `jax_available()` - Execute block if JAX is available
- `xarray_available()` - Execute block if xarray is available
- `pyarrow_available()` - Execute block if PyArrow is available
- `packaging_available()` - Execute block if packaging is available
- `torch_numpy_available()` - Execute block if both PyTorch and NumPy are available

### Raising Errors for Missing Packages

Use check functions to raise informative errors when required packages are missing:

```pycon

>>> from coola.utils.imports import check_numpy
>>> check_numpy()  # doctest: +SKIP
>>> # Raises RuntimeError with helpful message if NumPy is not installed

```

Available check functions (raise `RuntimeError` if package is missing):
- `check_numpy()` - Require NumPy
- `check_torch()` - Require PyTorch
- `check_pandas()` - Require pandas
- `check_polars()` - Require polars
- `check_jax()` - Require JAX
- `check_xarray()` - Require xarray
- `check_pyarrow()` - Require PyArrow
- `check_packaging()` - Require packaging
- `check_torch_numpy()` - Require both PyTorch and NumPy

### Decorator for Package Availability

Use `decorator_package_available()` to skip function execution if a package is not available:

```pycon
>>> from functools import partial
>>> from coola.utils.imports import decorator_package_available, is_numpy_available
>>> decorator = partial(decorator_package_available, condition=is_numpy_available)
>>> @decorator
... def process_with_numpy(data):
...     import numpy as np
...     return np.array(data).sum()
...
>>> # Function only executes if NumPy is available

```

### Lazy Imports

Use `lazy_import()` or `LazyModule` to defer imports until they are actually used:

```pycon

>>> from coola.utils.imports import lazy_import
>>> np = lazy_import("numpy")  # doctest: +SKIP
>>> # NumPy is not imported yet
>>> arr = np.array([1, 2, 3])  # doctest: +SKIP
>>> # NumPy is imported when first accessed

```

This is useful for:
- Reducing startup time by deferring expensive imports
- Making optional dependencies truly optional
- Avoiding circular import issues

## Formatting Utilities (`coola.utils.format`)

### Formatting Sequences

Use `str_sequence()` and `repr_sequence()` to format sequences:

```pycon

>>> from coola.utils.format import str_sequence, repr_sequence
>>> print(str_sequence([1, 2, 3, 4, 5]))
(0): 1
(1): 2
(2): 3
(3): 4
(4): 5
>>> print(repr_sequence([1, 2, 3, 4, 5]))
(0): 1
(1): 2
(2): 3
(3): 4
(4): 5

```

For single-line formatting, use `str_sequence_line()`:

```pycon

>>> from coola.utils.format import str_sequence_line
>>> str_sequence_line([1, 2, 3, 4, 5])
'1, 2, 3, 4, 5'

```

### Formatting Mappings

Use `str_mapping()` and `repr_mapping()` to format dictionaries:

```pycon

>>> from coola.utils.format import str_mapping, repr_mapping
>>> data = {"name": "Alice", "age": 30, "city": "NYC"}
>>> print(str_mapping(data))
(name): Alice
(age): 30
(city): NYC
>>> print(repr_mapping(data))
(name): Alice
(age): 30
(city): NYC

```

Sort keys before formatting:

```pycon

>>> from coola.utils.format import str_mapping
>>> data = {"z": 1, "a": 2, "m": 3}
>>> print(str_mapping(data, sorted_keys=True))
(a): 2
(m): 3
(z): 1

```

For single-line formatting, use `str_mapping_line()`:

```pycon

>>> from coola.utils.format import str_mapping_line
>>> str_mapping_line({"a": 1, "b": 2})
'a=1, b=2'

```

### Adding Indentation

Use `str_indent()` and `repr_indent()` to add indentation to multi-line strings:

```pycon

>>> from coola.utils.format import str_indent
>>> multi_line = "line1\nline2\nline3"
>>> print(str_indent(multi_line, num_spaces=4))
line1
    line2
    line3

```

### Human-Readable Byte Sizes

Use `str_human_byte_size()` to convert byte sizes to human-readable format:

```pycon

>>> from coola.utils.format import str_human_byte_size
>>> str_human_byte_size(512)
'512.00 B'
>>> str_human_byte_size(23456)
'22.91 KB'

```

Find the best byte unit for a size:

```pycon

>>> from coola.utils.format import find_best_byte_unit
>>> find_best_byte_unit(100)
'B'
>>> find_best_byte_unit(2000000)
'MB'

```

### Human-Readable Time Durations

Use `str_time_human()` to format time durations:

```pycon

>>> from coola.utils.format import str_time_human
>>> import datetime
>>> str_time_human(datetime.timedelta(seconds=90).seconds)
'0:01:30'
>>> str_time_human(datetime.timedelta(hours=2, minutes=30).seconds)
'2:30:00'

```

## Array Utilities (`coola.utils.array`)

The `coola.utils.array` module provides utilities for working with array-like objects across
different backends (NumPy, PyTorch).

## Tensor Utilities (`coola.utils.tensor`)

The `coola.utils.tensor` module provides utilities for working with PyTorch tensors:

- `is_cuda_available()` - Check if CUDA is available with PyTorch
- `is_mps_available()` - Check if MPS (Apple Silicon) is available with PyTorch

```pycon

>>> from coola.utils.tensor import is_cuda_available, is_mps_available
>>> is_cuda_available()  # doctest: +SKIP
False
>>> is_mps_available()  # doctest: +SKIP
False

```

## Mapping Utilities (`coola.utils.mapping`)

The `coola.utils.mapping` module provides utilities for working with mappings.

## Path Utilities (`coola.utils.path`)

The `coola.utils.path` module provides utilities for working with file paths.

## Version Utilities (`coola.utils.version`)

The `coola.utils.version` module provides utilities for working with package versions.

## Environment Variable Utilities (`coola.utils.env_vars`)

The `coola.utils.env_vars` module provides utilities for working with environment variables.

## Introspection Utilities (`coola.utils.introspection`)

The `coola.utils.introspection` module provides utilities for introspecting Python objects.

## Common Use Cases

### Conditional Imports in Libraries

Write library code that works with optional dependencies:

```pycon
>>> from collections.abc import Sequence
>>> from coola.utils.imports import is_numpy_available
>>> def process_data(data: Sequence[float]) -> float:
...     if is_numpy_available():
...         import numpy as np
...         return np.array(data).sum().item()
...     else:
...         return sum(data)
...
>>> process_data([1, 2, 3, 4, 5])
15

```

### Pretty Printing Debug Information

Format complex data structures for debugging:

```pycon

>>> from coola.utils.format import str_mapping
>>> config = {
...     "model": "transformer",
...     "layers": 12,
...     "hidden_size": 768,
...     "vocab_size": 50000,
... }
>>> print(str_mapping(config, sorted_keys=True))
(hidden_size): 768
(layers): 12
(model): transformer
(vocab_size): 50000

```

### Displaying File Sizes

Show file sizes in human-readable format:

```pycon

>>> from coola.utils.format import str_human_byte_size
>>> import os
>>> def show_file_size(filepath):  # doctest: +SKIP
...     size = os.path.getsize(filepath)
...     return f"{filepath}: {str_human_byte_size(size)}"
...

```

### Graceful Degradation

Provide fallback behavior when optional packages are not available:

```pycon

>>> from coola.utils.imports import numpy_available
>>> def compute_stats(data):
...     with numpy_available():  # doctest: +SKIP
...         # Use NumPy if available (faster)
...         import numpy as np
...         arr = np.array(data)
...         return {"mean": arr.mean(), "std": arr.std()}
...     # Fallback to pure Python
...     mean = sum(data) / len(data)
...     variance = sum((x - mean) ** 2 for x in data) / len(data)
...     return {"mean": mean, "std": variance**0.5}
...

```

## Design Principles

The `coola.utils` package follows these design principles:

1. **Modular**: Utilities are organized into focused submodules
2. **Efficient**: Uses caching where appropriate (e.g., import checks)
3. **Flexible**: Works with various backends and data types
4. **Informative**: Provides clear error messages and formatting
5. **Optional dependencies**: Gracefully handles missing packages

## See Also

- [`coola.testing`](testing.md): For pytest fixtures that use import utilities
- [`coola.reducer`](reducer.md): For examples of automatic backend selection
- [`coola.equality`](equality.md): For examples of handling multiple backends

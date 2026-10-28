# Utility Functions

:book: This page describes the `coola.utils` package, which provides the helpers that the other
`coola` packages are built on: imports, formatting, mappings, paths, and more.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.utils` package provides a collection of utility functions organized into several
submodules:

1. **Import utilities** - Check if packages are available and handle optional dependencies
2. **Other utilities** - Mapping, path, version, introspection, and statistics utilities

The functions that turn values into text are in [`coola.formatting`](formatting.md). The standalone helpers (array, tensor, Bloom filter, LRU cache, git, dates, timing, environment
variables, ...) are in [`coola.tools`](tools.md).

### Core helpers

The modules used by the other `coola` packages are the **core helpers**: `imports`, `fallback`,
`fork`, `integer`, `introspection`, `mapping`, `path`, `safe`, `selection`, `singleton`,
`stats`, `conversion` and `version`.

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
- `is_pydantic_available()` - Check if pydantic is available
- `is_rich_available()` - Check if rich is available
- `is_colorlog_available()` - Check if colorlog is available
- `is_torch_numpy_available()` - Check if both PyTorch and NumPy are available

### Decorators for Availability

`numpy_available`, `torch_available`, and the similar names for the other libraries are
**decorators**, not context managers. The decorated function is only defined if the package is
available, otherwise the name is bound to `None` (no error is raised):

```pycon
>>> from coola.utils.imports import numpy_available
>>> @numpy_available
... def total(data):
...     import numpy as np
...
...     return np.array(data).sum()
...

```

`with numpy_available():` raises a `TypeError`. To branch on availability, use
`is_numpy_available()` as in [Graceful Degradation](#graceful-degradation).

Available decorators: `numpy_available`, `torch_available`, `pandas_available`,
`polars_available`, `jax_available`, `xarray_available`, `pyarrow_available`,
`packaging_available`, `pydantic_available`, `rich_available`, `colorlog_available` and
`torch_numpy_available` (both PyTorch and NumPy).

### Raising Errors for Missing Packages

Use check functions to raise informative errors when required packages are missing:

```pycon
>>> from coola.utils.imports import check_numpy
>>> check_numpy()  # doctest: +SKIP
>>> # Raises MissingPackageError with helpful message if NumPy is not installed

```

Available check functions (raise `MissingPackageError` if the package is missing).
`MissingPackageError` is a subclass of both `ModuleNotFoundError` (so of `ImportError`) and
`RuntimeError`, so code that catches either keeps working:

- `check_numpy()` - Require NumPy
- `check_torch()` - Require PyTorch
- `check_pandas()` - Require pandas
- `check_polars()` - Require polars
- `check_jax()` - Require JAX
- `check_xarray()` - Require xarray
- `check_pyarrow()` - Require PyArrow
- `check_packaging()` - Require packaging
- `check_pydantic()` - Require pydantic
- `check_rich()` - Require rich
- `check_colorlog()` - Require colorlog
- `check_torch_numpy()` - Require both PyTorch and NumPy
- `check_package(package, command=None)` - Require any package. `command` is the install command
  added to the error message (by default, the message has no install command)

The `raise_<package>_missing_error()` functions (for example `raise_numpy_missing_error()`) raise the
same error unconditionally. They are used by the fallback classes and functions of
`coola.utils.fallback`, and `raise_package_missing_error(package_name, install_cmd)` does it for any
package.

### Decorator for Package Availability

Use `decorator_package_available()` to skip function execution if a package is not available:

```pycon
>>> from functools import partial
>>> from coola.utils.imports import decorator_package_available, is_numpy_available
>>> decorator = partial(decorator_package_available, condition=is_numpy_available)
>>> @decorator
... def process_with_numpy(data):
...     import numpy as np
...
...     return np.array(data).sum()
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

## Mapping Utilities (`coola.utils.mapping`)

The `coola.utils.mapping` module provides `sort_by_keys()` and `sort_by_values()`, which return a
new dictionary sorted by keys or by values:

```pycon
>>> from coola.utils.mapping import sort_by_keys, sort_by_values
>>> sort_by_keys({"b": 1, "a": 2})
{'a': 2, 'b': 1}
>>> sort_by_values({"a": 2, "b": 1})
{'b': 1, 'a': 2}

```

## Path Utilities (`coola.utils.path`)

The `coola.utils.path` module provides utilities for working with file paths:

- `sanitize_path()` - Convert a path to an absolute, normalized `Path` (it expands `~` and resolves
  `.` and `..`)
- `find_root_package_parent()` - Walk upward from a file or a directory to find the directory that
  contains the root Python package
- `working_directory()` - Context manager that temporarily changes the working directory
- `add_uuid_suffix()` - Return a path with a unique UUID added to the file name, before the
  extension

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.utils.path import add_uuid_suffix, sanitize_path, working_directory
>>> sanitize_path("~").is_absolute()
True
>>> path = add_uuid_suffix(Path("data.json"))
>>> path.name.startswith("data-") and path.suffix == ".json"
True
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     with working_directory(tmpdir):
...         Path.cwd() == Path(tmpdir).resolve()
True

```

## Version Utilities (`coola.utils.version`)

The `coola.utils.version` module provides utilities for working with package versions.
`get_package_version()` returns the installed version of a package (or `None` if the package is not
installed), and `compare_version()` compares it to a version with a comparison operator:

```pycon
>>> import operator
>>> from coola.utils.version import compare_version, get_package_version
>>> get_package_version("missing_package") is None
True
>>> compare_version("missing_package", operator.ge, "1.0")
False

```

## Introspection Utilities (`coola.utils.introspection`)

The `coola.utils.introspection` module provides utilities for introspecting Python objects:

- `get_fully_qualified_name()` - Fully qualified name of an object
- `get_type_constructor()` - Type of an object as a callable
- `get_all_child_classes()` - All the subclasses of a class, recursively
- `is_same_type()` - Indicate if an object has exactly the type of a reference (no subclass)
- `supports_methods()` - Indicate if an object has all the given callable methods
- `is_lambda_function()` and `check_not_lambda_function()` - Detect a lambda function

```pycon
>>> from pathlib import Path
>>> from coola.utils.introspection import (
...     get_all_child_classes,
...     get_fully_qualified_name,
...     is_same_type,
...     supports_methods,
... )
>>> get_fully_qualified_name(Path)
'pathlib.Path'
>>> is_same_type(True, 1)
False
>>> supports_methods([], "append", "pop")
True
>>> class A: ...
>>> class B(A): ...
>>> get_all_child_classes(A) == {B}
True

```

## Statistics Utilities (`coola.utils.stats`)

`quantile()` computes quantiles of a sequence of numbers with the linear method (the default of
`numpy.quantile`), without any dependency:

```pycon
>>> from coola.utils.stats import quantile
>>> quantile([1, 2, 3, 4, 5], [0.25, 0.5])
[2.0, 3.0]

```

## Common Use Cases

### Conditional Imports in Libraries

Write library code that works with optional dependencies:

```pycon
>>> from collections.abc import Sequence
>>> from coola.utils.imports import is_numpy_available
>>> def process_data(data: Sequence[float]) -> float:
...     if is_numpy_available():
...         import numpy as np
...
...         return np.array(data).sum().item()
...     else:
...         return sum(data)
>>> process_data([1, 2, 3, 4, 5])
15

```

### Pretty Printing Debug Information

Format complex data structures for debugging:

```pycon
>>> from coola.formatting.structure import str_mapping
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
>>> from coola.formatting.quantity import str_human_byte_size
>>> import os
>>> def show_file_size(filepath):  # doctest: +SKIP
...     size = os.path.getsize(filepath)
...     return f"{filepath}: {str_human_byte_size(size)}"

```

### Graceful Degradation

Provide fallback behavior when optional packages are not available:

```pycon
>>> from coola.utils.imports import is_numpy_available
>>> def compute_stats(data):
...     if is_numpy_available():
...         # Use NumPy if available (faster)
...         import numpy as np
...
...         arr = np.array(data)
...         return {"mean": arr.mean(), "std": arr.std()}
...     # Fallback to pure Python
...     mean = sum(data) / len(data)
...     variance = sum((x - mean) ** 2 for x in data) / len(data)
...     return {"mean": mean, "std": variance**0.5}
...

```

## Properties

| Property              | Statement                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------- |
| Optional dependencies | The import helpers report whether a package is available, without failing when it is not. |
| Cached checks         | The availability of a package is cached, so repeated checks are cheap.                    |
| Focused submodules    | Each submodule (`imports`, `mapping`, `path`, ...) has one responsibility.      |
| Informative errors    | Errors and formatted values are written to be read by a person.                           |

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

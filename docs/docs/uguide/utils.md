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

The functions that turn values into text are in [`coola.formatting`](formatting.md).

### Core helpers

The modules used by the other `coola` packages are the **core helpers**: `imports`, `fallback`,
`fork`, `integer`, `introspection`, `mapping`, `path`, `safe`, `selection`, `singleton`,
`stats` and `version`. The other modules are standalone helpers that `coola` itself does not
need (`array`, `tensor`, `batching`, `bloom_filter`, `date`, `env_vars`, `file_size`,
`git`, `logging`, `lru`, `password`, `recursion`, `timing`); they are public and tested like the rest:

| Module                     | Content                                                    |
| -------------------------- | ---------------------------------------------------------- |
| `coola.utils.array`        | Convert an object to a NumPy array                         |
| `coola.utils.tensor`       | Convert an object to a PyTorch tensor, device checks       |
| `coola.utils.batching`     | Split an iterable in batches                               |
| `coola.utils.bloom_filter` | `BloomFilter`, an approximate set for duplicate detection  |
| `coola.utils.date`         | Today's date and conversion of dates                       |
| `coola.utils.env_vars`     | Read, check and temporarily set environment variables      |
| `coola.utils.file_size`    | The size of a file or of a directory                       |
| `coola.utils.git`          | The branch and the commit of a git repository              |
| `coola.utils.logging`      | `CaptureHandler`, a logging handler that keeps the records |
| `coola.utils.lru`          | `LRUCache`, a least-recently-used cache                    |
| `coola.utils.password`     | Read a password from the terminal                          |
| `coola.utils.recursion`    | Explain a `RecursionError`                                 |
| `coola.utils.timing`       | The `timeblock` context manager                            |

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
- `check_package(package, *, install=None)` - Require any package. `install` is the name given to
  `pip install` when it differs from the import name

`raise_package_missing_error(package, *, install=None)` raises the same error unconditionally. All
the checks give the same message, which has the `pip install` and `uv pip install` commands. It is
used by the fallback classes and functions of `coola.utils.fallback`.

`is_<package>_available()` imports the package the first time it is called (a package that is
installed but broken is reported as not available) and caches the result. Use
`package_available(name)` to ask whether a package is installed without importing it.

### Lazy Imports

`lazy_exports()` defers the import of the names that a package exports until they are used, and
`is_module_imported()` indicates if the module of an object is already imported. To import an
object from a `"package.module:name"` string, use `pkgutil.resolve_name` of the standard library.

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
- `add_uuid_suffix()` - Return a path with a unique UUID added to the file name, before the
  extension

```pycon
>>> from pathlib import Path
>>> from coola.utils.path import add_uuid_suffix, sanitize_path
>>> sanitize_path("~").is_absolute()
True
>>> path = add_uuid_suffix(Path("data.json"))
>>> path.name.startswith("data-") and path.suffix == ".json"
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

- `get_qualified_class_name()` - Qualified name of a class, with its module unless it is a builtin
- `get_type_constructor()` - Type of an object as a callable
- `get_all_child_classes()` - All the subclasses of a class, recursively
- `is_same_type()` - Indicate if an object has exactly the type of a reference (no subclass)
- `supports_methods()` - Indicate if an object has all the given callable methods
- `is_lambda_function()` and `check_not_lambda_function()` - Detect a lambda function

```pycon
>>> from pathlib import Path
>>> from coola.utils.introspection import (
...     get_all_child_classes,
...     get_qualified_class_name,
...     is_same_type,
...     supports_methods,
... )
>>> get_qualified_class_name(Path)
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

## Array Utilities (`coola.utils.array`)

The `coola.utils.array` module provides utilities for working with array-like objects across
different backends (NumPy, PyTorch). `to_array()` converts a sequence of numbers, a
`numpy.ndarray` or a `torch.Tensor` to a `numpy.ndarray`:

```pycon
>>> from coola.utils.array import to_array
>>> to_array([1, 2, 3])
array([1, 2, 3])

```

## Tensor Utilities (`coola.utils.tensor`)

The `coola.utils.tensor` module provides utilities for working with PyTorch tensors:

- `is_cuda_available()` - Check if CUDA is available with PyTorch
- `is_mps_available()` - Check if MPS (Apple Silicon) is available with PyTorch
- `get_available_devices()` - Get the PyTorch devices available on the machine
- `to_tensor()` - Convert a sequence of numbers, a `numpy.ndarray` or a `torch.Tensor` to a
  `torch.Tensor`

```pycon
>>> from coola.utils.tensor import is_cuda_available, is_mps_available
>>> is_cuda_available()  # doctest: +SKIP
False
>>> is_mps_available()  # doctest: +SKIP
False

```

## Bloom Filter (`coola.utils.bloom_filter`)

The `coola.utils.bloom_filter` module provides `BloomFilter`, a fixed-memory approximate set
used for duplicate detection. It never reports a false negative, but it can report a false
positive at a configurable rate. It is a standalone utility and is not used by the other
`coola` packages.

```pycon
>>> from coola.utils.bloom_filter import BloomFilter
>>> bf = BloomFilter(expected_items=1000, fp_rate=0.01)
>>> bf.add_and_check(b"hello")
False
>>> bf.add_and_check(b"hello")
True

```

## Environment Variable Utilities (`coola.utils.env_vars`)

The `coola.utils.env_vars` module provides utilities for working with environment variables:

- `get_required_env_var()` - Get an environment variable and raise a `ValueError` if it is not set
  or is empty
- `check_env_vars()` - Indicate which environment variables are defined, and optionally raise or log
  at a given level when some are missing
- `temp_env_vars()` - Context manager that temporarily sets environment variables and restores them
  on exit

```pycon
>>> from coola.utils.env_vars import check_env_vars, get_required_env_var, temp_env_vars
>>> with temp_env_vars({"COOLA_EXAMPLE": "value"}):
...     get_required_env_var("COOLA_EXAMPLE")
...     check_env_vars(["COOLA_EXAMPLE"])
'value'
{'COOLA_EXAMPLE': True}
>>> get_required_env_var("COOLA_EXAMPLE")
Traceback (most recent call last):
    ...
ValueError: Environment variable 'COOLA_EXAMPLE' is required but not set or is empty

```

## Date Utilities (`coola.utils.date`)

The `coola.utils.date` module provides `get_today_date()` (today's date in a timezone, as a
string), `to_date()` and `to_datetime()` (convert a string to a `date` or a `datetime`):

```pycon
>>> from coola.utils.date import to_date, to_datetime
>>> to_date("2024-01-31")
datetime.date(2024, 1, 31)
>>> to_datetime("2024-01-31T10:00:00")
datetime.datetime(2024, 1, 31, 10, 0)

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

```

## Properties

`coola.utils` is the foundation of `coola`: the helpers that the other packages import. It is a leaf
of the package layering. The functions that turn values into text are in
[`coola.formatting`](formatting.md).

| Property                         | Statement                                                                                                                                                                                                            |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Leaf layer                       | `coola.utils` imports the standard library and `coola.validation` only. `coola.validation` does not import `coola.utils`. Both rules are checked by `lint-imports`.                                                  |
| Public, documented, tested       | A name used by another module has no leading underscore. Each public function has a Google docstring with a `pycon` example, a test module per source module, and a line in `refs/utils.md`.                         |
| One name per concept             | One concept has one function and one message: `get_qualified_class_name` is the one qualified name, and every missing package gives the same error message.                                                          |
| Optional packages stay optional  | `is_<pkg>_available()` is the single definition of "available" (the package can be imported). It is cached, and `clear_availability_caches()` resets it. `check_<pkg>()` raises `MissingPackageError`.               |
| Import-light                     | Importing a module of `coola.utils` imports no optional library and does no work, except `os.register_at_fork` in `fork`.                                                                                            |
| Strict contracts                 | An argument is checked by `coola.validation`, and options are keyword-only. A helper does not hide an error (`safe_str` and `safe_repr` report the error as a value, it is their job).                               |
| Stateless, with named exceptions | The exceptions are the availability caches, the registry of the objects that reset after a fork (`track_fork`) and the `LazySingleton` instances. No helper changes `os.environ`, the working directory or a logger. |
| Thread-safe, fork-safe           | A `LazySingleton` builds its value once under a lock, detects a request from its own factory, and rebuilds its lock in a forked child.                                                                               |
| Explicit modules                 | There is no re-export at `coola.utils`: import a name from the module that defines it (`coola.utils.mapping`). `coola.utils.imports` re-exports its public names.                                                    |

### Not Guaranteed

- **The availability of a package cannot change during the life of the process.** It is cached on
  the first call. A test that changes `sys.modules` calls `clear_availability_caches()`.
- **`is_<pkg>_available()` imports the package** the first time (a package that is installed but
  broken is reported as not available, with a logged warning). Use `package_available(name)` to ask
  whether a package is installed without importing it.
- **`LazySingleton` is not a registry of its values.** `reset()` forgets the value, and a reference
  obtained before is not changed.
- **The order of `smallest_items` when the items cannot be compared** is the iteration order of the
  collection.

## See Also

- [`coola.testing`](testing.md): For pytest fixtures that use import utilities
- [Formatting](formatting.md): the functions that turn values into text
- [`coola.reducer`](reducer.md): For examples of automatic backend selection
- [`coola.equality`](equality.md): For examples of handling multiple backends

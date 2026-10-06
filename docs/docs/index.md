# Home

<p align="center">
    <a href="https://github.com/durandtibo/coola/actions/workflows/ci.yaml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/durandtibo/coola/ci.yaml?branch=main&label=CI&logo=github"></a>
    <a href="https://github.com/durandtibo/coola/actions/workflows/nightly-package.yaml"><img alt="Nightly Package Tests" src="https://img.shields.io/github/actions/workflow/status/durandtibo/coola/nightly-package.yaml?label=nightly&logo=github"></a>
    <a href="https://codecov.io/gh/durandtibo/coola"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/durandtibo/coola?logo=codecov"></a>
    <br/>
    <a href="https://durandtibo.github.io/coola/"><img alt="Documentation (stable)" src="https://img.shields.io/badge/docs-stable-blue?logo=readthedocs&logoColor=white"></a>
    <a href="https://durandtibo.github.io/coola/dev/"><img alt="Documentation (unstable)" src="https://img.shields.io/badge/docs-dev-orange?logo=readthedocs&logoColor=white"></a>
    <br/>
    <a href="https://pypi.org/project/coola/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/coola?logo=pypi&logoColor=white"></a>
    <a href="https://pypi.org/project/coola/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/coola?logo=python&logoColor=white"></a>
    <a href="https://opensource.org/licenses/BSD-3-Clause"><img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/coola"></a>
    <br/>
    <a href="https://pepy.tech/project/coola"><img alt="Downloads" src="https://img.shields.io/pepy/dt/coola"></a>
    <a href="https://pepy.tech/project/coola"><img alt="Monthly downloads" src="https://img.shields.io/pepy/dm/coola"></a>
    <a href="https://github.com/durandtibo/coola/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/durandtibo/coola?logo=github"></a>
    <br/>
    <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings"><img alt="Doc style: google" src="https://img.shields.io/badge/docstyle-google-3666d6"></a>
    <a href="https://www.bestpractices.dev/projects/14387"><img alt="OpenSSF Best Practices" src="https://img.shields.io/cii/summary/14387?label=openssf%20best%20practices"></a>
    <a href="https://scorecard.dev/viewer/?uri=github.com/durandtibo/coola"><img alt="OpenSSF Scorecard" src="https://img.shields.io/ossf-scorecard/github.com/durandtibo/coola?label=openssf%20scorecard"></a>
</p>

## Overview

`coola` is a lightweight Python library that makes it easy to compare complex and nested data
structures.
It provides simple, extensible functions to check equality between objects containing
[PyTorch tensors](https://pytorch.org/docs/stable/tensors.html),
[NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html),
[pandas](https://pandas.pydata.org/)/[polars](https://www.pola.rs/) DataFrames, and other scientific
computing objects.

**Quick Links:**

- [User Guide](uguide/equality.md)
- [Installation](get_started.md)
- [Features](#features)
- [Contributing](#contributing)

## Why coola?

Python's native equality operator (`==`) doesn't work well with complex nested structures
containing tensors, arrays, or DataFrames. You'll often encounter errors or unexpected behavior.
`coola` solves this with intuitive comparison functions:

**Check exact equality:**

```pycon
>>> import numpy as np
>>> import torch
>>> from coola.equality import objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": np.zeros((2, 3))}
>>> data2 = {"torch": torch.ones(2, 3), "numpy": np.zeros((2, 3))}
>>> objects_are_equal(data1, data2)
True

```

**Compare with numerical tolerance:**

```pycon
>>> from coola.equality import objects_are_allclose
>>> data1 = {"value": 1.0}
>>> data2 = {"value": 1.0 + 1e-9}
>>> objects_are_allclose(data1, data2)
True

```

See the [user guide](uguide/equality.md) for detailed examples.

## Features

`coola` provides a comprehensive set of utilities for working with complex data structures:

### 🔍 **Equality Comparison**

Compare complex nested objects with support for multiple data types:

- **Exact equality**: `objects_are_equal()` for strict comparison
- **Approximate equality**: `objects_are_allclose()` for numerical tolerance
- **User-friendly difference reporting**: Clear, structured output showing exactly what differs
- **Extensible**: Add custom equality testers for your own types

[Learn more →](uguide/equality.md)

**Supported types:**
[JAX](https://jax.readthedocs.io/) •
[NumPy](https://numpy.org/) •
[pandas](https://pandas.pydata.org/) •
[polars](https://www.pola.rs/) •
[PyArrow](https://arrow.apache.org/docs/python/) •
[PyTorch](https://pytorch.org/) •
[xarray](https://docs.xarray.dev/) •
Python built-ins (dict, list, tuple, set, etc.)

[Learn more about supported types →](uguide/equality.md#type-specific-behavior)

### 📊 **Data Summarization**

Generate human-readable summaries of nested data structures for debugging and logging:

- Configurable depth control
- Type-specific formatting
- Truncation for large collections

[Learn more →](uguide/summary.md)

### 🔄 **Data Conversion**

Transform data between different nested structures:

- Convert between list-of-dicts and dict-of-lists formats
- Useful for working with tabular data and different data representations

[Learn more →](uguide/nested.md)

### 🗂️ **Mapping Utilities**

Work with nested dictionaries efficiently:

- Flatten nested dictionaries into flat key-value pairs
- Extract specific values from complex nested structures
- Filter dictionary keys based on patterns or criteria

[Learn more →](uguide/nested.md)

### 🔁 **Iteration**

Traverse nested data structures systematically:

- Depth-first search (DFS) traversal for nested containers
- Breadth-first search (BFS) traversal for level-by-level processing
- Filter and extract specific types from heterogeneous collections

[Learn more →](uguide/iterator.md)

### 📈 **Reduction**

Compute statistics on sequences with flexible backends:

- Calculate min, max, mean, median, quantile, std on numeric sequences
- Support for multiple backends: native Python, NumPy, PyTorch
- Consistent API regardless of backend choice

[Learn more →](uguide/reducer.md)

## Contributing

Contributions are welcome! We appreciate bug fixes, feature additions, documentation improvements,
and more. Please check
the [contributing guidelines](https://github.com/durandtibo/coola/blob/main/CONTRIBUTING.md) for
details on:

- Setting up the development environment
- Code style and testing requirements
- Submitting pull requests

Whether you're fixing a bug or proposing a new feature, please open an issue first to discuss
your changes.

## API Stability

:warning: **Important**: As `coola` is under active development, its API is not yet stable and may
change between releases. We recommend pinning a specific version in your project’s dependencies to
ensure consistent behavior.

## License

`coola` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](https://github.com/durandtibo/coola/blob/main/LICENSE)
file.

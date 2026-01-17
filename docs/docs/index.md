# Home

<p align="center">
    <a href="https://github.com/durandtibo/coola/actions/workflows/ci.yaml">
        <img alt="CI" src="https://github.com/durandtibo/coola/actions/workflows/ci.yaml/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/coola/actions/workflows/nightly-tests.yaml">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/coola/actions/workflows/nightly-tests.yaml/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/coola/actions/workflows/nightly-package.yaml">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/coola/actions/workflows/nightly-package.yaml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/durandtibo/coola">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/coola/branch/main/graph/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/coola/">
        <img alt="Documentation" src="https://github.com/durandtibo/coola/actions/workflows/docs.yaml/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/coola/dev/">
        <img alt="Documentation" src="https://github.com/durandtibo/coola/actions/workflows/docs-dev.yaml/badge.svg">
    </a>
    <br/>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;">
    </a>
    <a href="https://github.com/guilatrova/tryceratops">
        <img  alt="try/except style: tryceratops" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
    </a>
    <br/>
    <a href="https://pypi.org/project/coola/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/coola">
    </a>
    <a href="https://pypi.org/project/coola/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/coola.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/coola">
    </a>
    <br/>
    <a href="https://pepy.tech/project/coola">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/coola">
    </a>
    <a href="https://pepy.tech/project/coola">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/coola/month">
    </a>
    <br/>
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

- [Quickstart](uguide/quickstart.md)
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

See the [quickstart guide](uguide/quickstart.md) for detailed examples.

## Features

`coola` provides a comprehensive set of utilities for working with complex data structures:

### 🔍 **Equality Comparison**

Compare complex nested objects with support for multiple data types:

- **Exact equality**: `objects_are_equal()` for strict comparison
- **Approximate equality**: `objects_are_allclose()` for numerical tolerance
- **Extensible**: Add custom comparators for your own types

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

[See the full list of supported types →](uguide/equality_types.md)

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

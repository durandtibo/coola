# coola

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
        <img  alt="Doc style: google" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
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

`coola` is a lightweight Python library for comparing complex and nested data structures.
It provides simple, extensible functions to check equality between objects containing
[PyTorch tensors](https://pytorch.org/docs/stable/tensors.html),
[NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html),
pandas/polars DataFrames, and other scientific computing objects.

**Quick Links:**
- [Documentation](https://durandtibo.github.io/coola/) | [Quickstart](https://durandtibo.github.io/coola/quickstart)
- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license) | [Security](SECURITY.md)

## Why coola?

Python's native equality operator `==` doesn't work well with complex nested structures
containing tensors, arrays, or DataFrames. `coola` solves this with simple comparison functions:

```pycon
>>> import numpy as np
>>> import torch
>>> from coola.equality import objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": np.zeros((2, 3))}
>>> data2 = {"torch": torch.ones(2, 3), "numpy": np.zeros((2, 3))}
>>> objects_are_equal(data1, data2)
True
```

For numerical comparisons with tolerance:

```pycon
>>> from coola.equality import objects_are_allclose
>>> data1 = {"value": 1.0}
>>> data2 = {"value": 1.0 + 1e-9}
>>> objects_are_allclose(data1, data2)
True
```

See the [quickstart guide](https://durandtibo.github.io/coola/quickstart) for detailed examples.

## Features

`coola` provides a comprehensive set of utilities for working with complex data structures:

### 🔍 **Equality Comparison**
Compare complex nested objects with support for multiple data types.
- **Exact equality**: `objects_are_equal()` for strict comparison
- **Approximate equality**: `objects_are_allclose()` for numerical tolerance
- **Extensible**: Add custom comparators for your own types
- [Learn more →](https://durandtibo.github.io/coola/uguide/equality)

**Supported types:**
[JAX](https://jax.readthedocs.io/) •
[NumPy](https://numpy.org/) •
[pandas](https://pandas.pydata.org/) •
[polars](https://www.pola.rs/) •
[PyArrow](https://arrow.apache.org/docs/python/) •
[PyTorch](https://pytorch.org/) •
[xarray](https://docs.xarray.dev/) •
Python built-ins (dict, list, tuple, set, etc.)

See the [full list of supported types →](https://durandtibo.github.io/coola/uguide/equality_types)

### 📊 **Data Summarization**
Generate human-readable summaries of nested data structures for debugging and logging.
- Configurable depth control
- Type-specific formatting
- Truncation for large collections
- [Learn more →](https://durandtibo.github.io/coola/uguide/summary)

### 🧪 **Testing Utilities**
Pytest fixtures and markers for handling optional dependencies in your tests.
- [Learn more →](https://durandtibo.github.io/coola/uguide/testing)

### 🔧 **Additional Utilities**
- **Nested operations**: Work with deeply nested data structures ([docs](https://durandtibo.github.io/coola/uguide/nested))
- **Iterators**: Traverse nested structures ([docs](https://durandtibo.github.io/coola/uguide/iterator))
- **Reducers**: Aggregate values in nested structures ([docs](https://durandtibo.github.io/coola/uguide/reducer))
- **Random utilities**: Generate random data structures ([docs](https://durandtibo.github.io/coola/uguide/random))
- **Recursive utilities**: Apply functions recursively ([docs](https://durandtibo.github.io/coola/uguide/recursive))

## Installation

We highly recommend installing in a
[virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

### Using pip (recommended)

```shell
pip install coola
```

This installs the minimal required dependencies. To include all optional dependencies:

```shell
pip install coola[all]
```

Or install specific optional dependencies:

```shell
pip install coola  # minimal installation
pip install coola[numpy,torch]  # with NumPy and PyTorch
```

### Requirements

- **Python**: 3.10 or higher
- **Dependencies**: No required dependencies (all are optional)

**Optional dependencies** (installed with `coola[all]`):
[JAX](https://jax.readthedocs.io/) •
[NumPy](https://numpy.org/) •
[pandas](https://pandas.pydata.org/) •
[polars](https://www.pola.rs/) •
[PyArrow](https://arrow.apache.org/docs/python/) •
[PyTorch](https://pytorch.org/) •
[xarray](https://docs.xarray.dev/)

For detailed installation instructions, compatibility information, and alternative installation
methods, see the [installation guide](https://durandtibo.github.io/coola/get_started).

## Contributing

Contributions are welcome! Please check the [contributing guidelines](CONTRIBUTING.md) for details on:
- Setting up the development environment
- Code style and testing requirements
- Submitting pull requests

## API Stability

:warning: While `coola` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `coola` to a new version will possibly break any code that
was using the old version of `coola`.

## License

`coola` is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.

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

### 🔄 **Data Conversion**

Transform data between different nested structures.

- Convert between list-of-dicts and dict-of-lists formats
- Useful for working with tabular data and different data representations
- [Learn more →](https://durandtibo.github.io/coola/uguide/nested)

### 🗂️ **Mapping Utilities**

Work with nested dictionaries efficiently.

- Flatten nested dictionaries into flat key-value pairs
- Extract specific values from complex nested structures
- Filter dictionary keys based on patterns or criteria
- [Learn more →](https://durandtibo.github.io/coola/uguide/nested)

### 🔁 **Iteration**

Traverse nested data structures systematically.

- Depth-first search (DFS) traversal for nested containers
- Breadth-first search (BFS) traversal for level-by-level processing
- Filter and extract specific types from heterogeneous collections
- [Learn more →](https://durandtibo.github.io/coola/uguide/iterator)

### 📈 **Reduction**

Compute statistics on sequences with flexible backends.

- Calculate min, max, mean, median, quantile, std on numeric sequences
- Support for multiple backends: native Python, NumPy, PyTorch
- Consistent API regardless of backend choice
- [Learn more →](https://durandtibo.github.io/coola/uguide/reducer)

## Installation

We highly recommend installing in a
[virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

### Using uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package installer and resolver:

```shell
uv pip install coola
```

To include all optional dependencies:

```shell
uv pip install coola[all]
```

Or install specific optional dependencies:

```shell
uv pip install coola[numpy,torch]  # with NumPy and PyTorch
```

### Using pip

Alternatively, you can use `pip`:

```shell
pip install coola
```

To include all optional dependencies:

```shell
pip install coola[all]
```

Or install specific optional dependencies:

```shell
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
The following is the corresponding `coola` versions and tested dependencies.

| `coola`  | `jax`<sup>*</sup> | `numpy`<sup>*</sup> | `packaging`<sup>*</sup> | `pandas`<sup>*</sup> | `polars`<sup>*</sup> | `pyarrow`<sup>*</sup> | `torch`<sup>*</sup> | `xarray`<sup>*</sup> | `python`       |
|----------|-------------------|---------------------|-------------------------|----------------------|----------------------|-----------------------|---------------------|----------------------|----------------|
| `main`   | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10`       |
| `0.11.1` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10`       |
| `0.11.0` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10`       |
| `0.10.0` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10`       |
| `0.9.1`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0,<26.0`          | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10,<3.15` |
| `0.9.0`  | `>=0.4.6,<1.0`    | `>=1.24,<3.0`       | `>=22.0,<26.0`          | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<20.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.9,<3.14`  |

<sup>*</sup> indicates an optional dependency

<details>
    <summary>older versions</summary>

| `coola`  | `jax`<sup>*</sup> | `numpy`<sup>*</sup> | `packaging`<sup>*</sup> | `pandas`<sup>*</sup> | `polars`<sup>*</sup> | `pyarrow`<sup>*</sup> | `torch`<sup>*</sup> | `xarray`<sup>*</sup> | `python`      |
|----------|-------------------|---------------------|-------------------------|----------------------|----------------------|-----------------------|---------------------|----------------------|---------------|
| `0.8.7`  | `>=0.4.6,<1.0`    | `>=1.22,<3.0`       | `>=21.0,<26.0`          | `>=1.5,<3.0`         | `>=1.0,<2.0`         | `>=10.0,<20.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14` |
| `0.8.6`  | `>=0.4.6,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<20.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14` |
| `0.8.5`  | `>=0.4.6,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<19.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14` |
| `0.8.4`  | `>=0.4.6,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14` |
| `0.8.3`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.8.2`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.8.1`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.8.0`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.4`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.3`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      |                       | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.2`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      |                       | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.1`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.0`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.6.2`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.6.1`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.6.0`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.5.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.4.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.3.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.3.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.2.2`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.2.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.2.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.1.2`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.21`     |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.1.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.1.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.12` |
| `0.0.26` | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.12` |
| `0.0.25` | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.4,<2023.11`  | `>=3.9,<3.12` |
| `0.0.24` | `>=0.3,<0.5`      | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12` |
| `0.0.23` | `>=0.3,<0.5`      | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.1`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12` |
| `0.0.22` | `>=0.3,<0.5`      | `>=1.20,<1.26`      |                         | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     |                       | `>=1.10,<2.1`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12` |
| `0.0.21` | `>=0.3,<0.5`      | `>=1.20,<1.26`      |                         | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     |                       | `>=1.10,<2.1`       | `>=2023.3,<2023.8`   | `>=3.9,<3.12` |
| `0.0.20` | `>=0.3,<0.5`      | `>=1.20,<1.26`      |                         | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     |                       | `>=1.10,<2.1`       | `>=2023.3,<2023.8`   | `>=3.9`       |

</details>

## Contributing

Contributions are welcome! Please check the [contributing guidelines](CONTRIBUTING.md) for details
on:

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

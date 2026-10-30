# coola

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

`coola` is a lightweight Python library to **compare**, **hash** and **summarize** complex and
nested data structures.
It provides simple, extensible functions that work on objects containing
[PyTorch tensors](https://pytorch.org/docs/stable/tensors.html),
[NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html),
[pandas](https://pandas.pydata.org/)/[polars](https://www.pola.rs/) DataFrames, and other scientific
computing objects:

- **Compare**: `objects_are_equal`, `objects_are_allclose`, `assert_objects_equal` and `compare`
  check equality (exact or with a tolerance) and tell where two objects differ.
- **Hash**: `hash_object` computes a deterministic hash of a nested object.
- **Summarize**: `summarize` creates a short, readable description of a large structure for logs.

**Quick Links:**

- [Documentation](https://durandtibo.github.io/coola/)
- [User Guide](https://durandtibo.github.io/coola/uguide/equality)
- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Security](SECURITY.md)

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

**Assert equality in your tests, with the path of the difference:**

```pycon
>>> from coola.equality import assert_objects_equal
>>> actual = {"users": [{"id": 1, "score": 95}, {"id": 2, "score": 87}]}
>>> expected = {"users": [{"id": 1, "score": 95}, {"id": 2, "score": 88}]}
>>> assert_objects_equal(actual, expected)
Traceback (most recent call last):
    ...
AssertionError: objects are not equal at actual['users'][1]['score']
numbers are different:
  actual   : 87
  expected : 88

```

`assert_objects_allclose` does the same with a tolerance.

**Get the difference as data:**

```pycon
>>> from coola.equality import compare
>>> result = compare(actual, expected)
>>> result.equal
False
>>> result.path
('users', 1, 'score')
>>> print(result.reason)
numbers are different:
  actual   : 87
  expected : 88

```

**Hash and summarize:**

```pycon
>>> from coola.hashing import hash_object
>>> from coola.summary import summarize
>>> len(hash_object(actual))
64
>>> print(summarize(actual, max_depth=2))
<class 'dict'> (length=1)
  (users): <class 'list'> (length=2)
      (0): {'id': 1, 'score': 95}
      (1): {'id': 2, 'score': 87}

```

`compare(actual, expected)` explains the first difference (its path and its reason), and
`assert_objects_equal(actual, expected)` raises an `AssertionError` with the same message.

See the [user guide](https://durandtibo.github.io/coola/uguide/equality) for detailed examples.

## Features

`coola` provides a comprehensive set of utilities for working with complex data structures:

### 🔍 **Equality Comparison**

Compare complex nested objects with support for multiple data types:

- **Exact equality**: `objects_are_equal()` for strict comparison
- **Approximate equality**: `objects_are_allclose()` for numerical tolerance
- **Assertions for tests**: `assert_objects_equal()` and `assert_objects_allclose()` raise an
  `AssertionError` with the path of the difference
- **Structured results**: `compare()` returns a `ComparisonResult` with the path and the reason
- **Extensible**: Add custom equality testers for your own types

[Learn more →](https://durandtibo.github.io/coola/uguide/equality)

**Supported types:**
[JAX](https://jax.readthedocs.io/) •
[NumPy](https://numpy.org/) •
[pandas](https://pandas.pydata.org/) •
[polars](https://www.pola.rs/) •
[PyArrow](https://arrow.apache.org/docs/python/) •
[pydantic](https://docs.pydantic.dev/) •
[PyTorch](https://pytorch.org/) •
[SciPy](https://scipy.org/) (sparse matrices and arrays) •
[xarray](https://docs.xarray.dev/) •
Python built-ins (dict, list, tuple, set, etc.)

[See all type-specific comparison rules →](https://durandtibo.github.io/coola/uguide/equality#type-specific-behavior)

### 📊 **Data Summarization**

Generate human-readable summaries of nested data structures for debugging and logging:

- Configurable depth control
- Type-specific formatting
- Truncation for large collections

[Learn more →](https://durandtibo.github.io/coola/uguide/summary)

### 🔄 **Data Conversion**

Transform data between different nested structures:

- Convert between list-of-dicts and dict-of-lists formats
- Useful for working with tabular data and different data representations

[Learn more →](https://durandtibo.github.io/coola/uguide/nested)

### 🗂️ **Mapping Utilities**

Work with nested dictionaries efficiently:

- Flatten nested dictionaries into flat key-value pairs
- Extract specific values from complex nested structures
- Filter dictionary keys based on patterns or criteria, rename keys and merge mappings

[Learn more →](https://durandtibo.github.io/coola/uguide/nested)

### 🔁 **Iteration**

Traverse nested data structures systematically:

- Depth-first search (DFS) traversal for nested containers
- Breadth-first search (BFS) traversal for level-by-level processing
- Filter and extract specific types from heterogeneous collections

[Learn more →](https://durandtibo.github.io/coola/uguide/iterator)

### 📈 **Reduction**

Compute statistics on sequences with flexible backends:

- Calculate min, max, mean, median, quantile, std on numeric sequences
- Support for multiple backends: native Python, NumPy, PyTorch
- Consistent API regardless of backend choice

[Learn more →](https://durandtibo.github.io/coola/uguide/reducer)

### #️⃣ **Hashing**

Compute stable hashes of nested objects:

- Hash dicts, lists, sets, dataclasses, pydantic models, arrays and data frames
- Independent of the insertion order of mappings and sets
- Extensible with custom hashers

[Learn more →](https://durandtibo.github.io/coola/uguide/hashing)

### 🆔 **Identifiers**

Generate unique identifiers: ULID, UUID (v5, v7), Snowflake, ObjectId, and checksummed and
content-based identifiers.

[Learn more →](https://durandtibo.github.io/coola/uguide/identifier)

### 💾 **Data Loading and Saving**

Load and save JSON, pickle, text and PyTorch files with atomic writes, an overwrite policy and
optional durability (`fsync`).

[Learn more →](https://durandtibo.github.io/coola/uguide/io)

### 🎲 **Random Number Generators**

Seed, save and restore the random number generators of `random`, NumPy and PyTorch in one call.

[Learn more →](https://durandtibo.github.io/coola/uguide/random)

### 🔧 **Recursive Transformation and Validation**

- Apply a function to every leaf of a nested structure
- Validate arguments with clear error messages

[Learn more about transformation →](https://durandtibo.github.io/coola/uguide/recursive) · [Learn more about validation →](https://durandtibo.github.io/coola/uguide/validation)

### 🧩 **Registries and Factories**

Build extensible components:

- Thread-safe `Registry` and `TypeRegistry` (lookup by Method Resolution Order, lazy loading of
  optional libraries)
- Build any importable class or function from a configuration dictionary with `coola.factory`

[Learn more about registries →](https://durandtibo.github.io/coola/uguide/registry) · [Learn more about factories →](https://durandtibo.github.io/coola/uguide/factory)

### 🧾 **JSON Conversion**

Convert nested data (dataclasses, sets, paths, arrays, tensors, pydantic models, ...) to JSON data
with `to_jsonable`, with a strict, extensible and cycle-safe conversion.

[Learn more →](https://durandtibo.github.io/coola/uguide/jsonable)

### 🔢 **Rounding**

Round, floor or ceil numbers, NumPy arrays, PyTorch tensors, and pandas, polars and PyArrow data
with `round_values`.

[Learn more →](https://durandtibo.github.io/coola/uguide/rounding)

### 🖨️ **Display and Formatting**

Build consistent `__repr__`/`__str__` with display mixins, mask secrets, format sizes and durations,
and configure `rich` and `colorlog` output.

[Learn more about display →](https://durandtibo.github.io/coola/uguide/display) · [Learn more about formatting →](https://durandtibo.github.io/coola/uguide/formatting)

### 🧪 **Testing and Utilities**

`pytest` marks and fixtures to test code with optional dependencies, and helpers for imports, paths,
environment variables, caches and more.

[Learn more about testing →](https://durandtibo.github.io/coola/uguide/testing) · [Learn more about utilities →](https://durandtibo.github.io/coola/uguide/utils)

## Installation

We highly recommend installing
`coola` in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to avoid dependency conflicts.

### Using uv (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package installer and resolver:

```shell
uv pip install coola
```

**Install with several optional dependencies:**

```shell
uv pip install coola[jax,numpy,packaging,pandas,polars,pyarrow,pydantic,scipy,torch,xarray,testing,colorlog,rich]
```

**Install with specific optional dependencies:**

```shell
uv pip install coola[numpy,torch]  # with NumPy and PyTorch
```

### Using pip

Alternatively, you can use `pip`:

```shell
pip install coola
```

**Install with several optional dependencies:**

```shell
pip install coola[jax,numpy,packaging,pandas,polars,pyarrow,pydantic,scipy,torch,xarray,testing,colorlog,rich]
```

**Install with specific optional dependencies:**

```shell
pip install coola[numpy,torch]  # with NumPy and PyTorch
```

### Requirements

- **Python**: 3.11 or higher
- **Operating system**: Linux or macOS (no Windows support)
- **Core dependencies**: None (fully optional dependencies)

**Optional dependencies** (install with `coola[<name>]`, e.g. `coola[numpy,torch]`):
[JAX](https://jax.readthedocs.io/) •
[NumPy](https://numpy.org/) •
[pandas](https://pandas.pydata.org/) •
[polars](https://www.pola.rs/) •
[PyArrow](https://arrow.apache.org/docs/python/) •
[PyTorch](https://pytorch.org/) •
[SciPy](https://scipy.org/) •
[xarray](https://docs.xarray.dev/) •
[pydantic](https://docs.pydantic.dev/) •
[packaging](https://packaging.pypa.io/) •
[colorlog](https://pypi.org/project/colorlog/) •
[rich](https://rich.readthedocs.io/)

> **Note:** the `jax` and `torch` extras are further restricted to Linux and Apple Silicon (macOS
> arm64), matching the platforms these libraries publish wheels for — installing `coola[jax]` or
> `coola[torch]` on Intel macOS will fail to resolve.

For detailed installation instructions, compatibility information, and alternative installation
methods, see the [installation guide](https://durandtibo.github.io/coola/get_started).

### Compatibility Matrix

| `coola`  | `jax`<sup>*</sup> | `numpy`<sup>*</sup> | `packaging`<sup>*</sup> | `pandas`<sup>*</sup> | `polars`<sup>*</sup> | `pyarrow`<sup>*</sup> | `pydantic`<sup>*</sup> | `scipy`<sup>*</sup> | `torch`<sup>*</sup> | `xarray`<sup>*</sup> | `python` |
| -------- | ----------------- | ------------------- | ----------------------- | -------------------- | -------------------- | --------------------- | ---------------------- | ------------------- | ------------------- | -------------------- | -------- |
| `main`   | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<3.0`         | `>=11.0`              | `>=2.0,<3.0`           | `>=1.13`            | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.11` |
| `1.1.10` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              | `>=2.0,<3.0`           |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.9`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              | `>=2.0,<3.0`           |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.8`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.7`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.6`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.5`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.4`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.3`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.2`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.1`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.1.0`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.0.1`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<4.0`         | `>=1.0,<2.0`         | `>=11.0`              |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |
| `1.0.0`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<23.0`        |                        |                     | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10` |

<sup>*</sup> indicates an optional dependency

<details>
    <summary>older versions</summary>

| `coola`  | `jax`<sup>*</sup> | `numpy`<sup>*</sup> | `packaging`<sup>*</sup> | `pandas`<sup>*</sup> | `polars`<sup>*</sup> | `pyarrow`<sup>*</sup> | `torch`<sup>*</sup> | `xarray`<sup>*</sup> | `python`       |
| -------- | ----------------- | ------------------- | ----------------------- | -------------------- | -------------------- | --------------------- | ------------------- | -------------------- | -------------- |
| `0.11.1` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2024.1`           | `>=3.10`       |
| `0.11.0` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10`       |
| `0.10.0` | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0`                | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10`       |
| `0.9.1`  | `>=0.5.0,<1.0`    | `>=1.24,<3.0`       | `>=22.0,<26.0`          | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<22.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.10,<3.15` |
| `0.9.0`  | `>=0.4.6,<1.0`    | `>=1.24,<3.0`       | `>=22.0,<26.0`          | `>=2.0,<3.0`         | `>=1.0,<2.0`         | `>=11.0,<20.0`        | `>=2.0,<3.0`        | `>=2023.1`           | `>=3.9,<3.14`  |
| `0.8.7`  | `>=0.4.6,<1.0`    | `>=1.22,<3.0`       | `>=21.0,<26.0`          | `>=1.5,<3.0`         | `>=1.0,<2.0`         | `>=10.0,<20.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14`  |
| `0.8.6`  | `>=0.4.6,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<20.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14`  |
| `0.8.5`  | `>=0.4.6,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<19.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14`  |
| `0.8.4`  | `>=0.4.6,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14`  |
| `0.8.3`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.8.2`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.8.1`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.8.0`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.7.4`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.7.3`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      |                       | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.7.2`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      |                       | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.7.1`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.7.0`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.6.2`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.6.1`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.6.0`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       |                         | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13`  |
| `0.5.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.4.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.3.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.3.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.2.2`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.2.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.2.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.1.2`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.21`     |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.1.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13`  |
| `0.1.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.12`  |
| `0.0.26` | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.12`  |
| `0.0.25` | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.4,<2023.11`  | `>=3.9,<3.12`  |
| `0.0.24` | `>=0.3,<0.5`      | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.2`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12`  |
| `0.0.23` | `>=0.3,<0.5`      | `>=1.21,<1.27`      |                         | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     |                       | `>=1.10,<2.1`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12`  |
| `0.0.22` | `>=0.3,<0.5`      | `>=1.20,<1.26`      |                         | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     |                       | `>=1.10,<2.1`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12`  |
| `0.0.21` | `>=0.3,<0.5`      | `>=1.20,<1.26`      |                         | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     |                       | `>=1.10,<2.1`       | `>=2023.3,<2023.8`   | `>=3.9,<3.12`  |
| `0.0.20` | `>=0.3,<0.5`      | `>=1.20,<1.26`      |                         | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     |                       | `>=1.10,<2.1`       | `>=2023.3,<2023.8`   | `>=3.9`        |

</details>

## Contributing

Contributions are welcome! We appreciate bug fixes, feature additions, documentation improvements,
and more. Please check the [contributing guidelines](CONTRIBUTING.md) for details on:

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

`coola` is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.

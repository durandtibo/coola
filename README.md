# coola

<p align="center">
    <a href="https://github.com/durandtibo/coola/actions">
        <img alt="CI" src="https://github.com/durandtibo/coola/workflows/CI/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/coola/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/coola/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/coola/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/coola/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/coola/">
        <img alt="Documentation" src="https://github.com/durandtibo/coola/workflows/Documentation%20(stable)/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/coola/">
        <img alt="Documentation" src="https://github.com/durandtibo/coola/workflows/Documentation%20(unstable)/badge.svg">
    </a>
    <br/>
    <a href="https://codecov.io/gh/durandtibo/coola">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/coola/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/coola/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/83ebb50e6c6f67b0570d/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/coola/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/83ebb50e6c6f67b0570d/test_coverage" />
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

`coola` is a Python library that provides simple functions to check in a single line if two
complex/nested objects are equal or not.
`coola` was initially designed to work
with [PyTorch `Tensor`s](https://pytorch.org/docs/stable/tensors.html)
and [NumPy `ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html), but it
is possible to extend it
to [support other data structures](https://durandtibo.github.io/coola/customization).

- [Motivation](#motivation)
- [Documentation](https://durandtibo.github.io/coola/)
- [Installation](#installation)
- [Contributing](#contributing)
- [API stability](#api-stability)
- [License](#license)

## Motivation

Let's imagine you have the following dictionaries that contain both a PyTorch `Tensor` and a
NumPy `ndarray`.
You want to check if the two dictionaries are equal or not.
By default, Python does not provide an easy way to check if the two dictionaries are equal or not.
It is not possible to use the default equality operator `==` because it will raise an error.
The `coola` library was developed to fill this gap. `coola` provides a function `objects_are_equal`
that can indicate if two complex/nested objects are equal or not.

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> objects_are_equal(data1, data2)
False

```

`coola` also provides a function `objects_are_allclose` that can indicate if two complex/nested
objects are equal within a tolerance or not.

```pycon

>>> import numpy
>>> import torch
>>> from coola import objects_are_allclose
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> objects_are_allclose(data1, data2, atol=1e-6)
False

```

`coola` supports the following types:

- [`jax.numpy.ndarray`](https://jax.readthedocs.io/en/latest/index.html)
- [`numpy.ndarray`](https://numpy.org/doc/stable/index.html)
- [`numpy.ma.MaskedArray`](https://numpy.org/doc/stable/reference/maskedarray.generic.html)
- [`pandas.DataFrame`](https://pandas.pydata.org/)
- [`pandas.Series`](https://pandas.pydata.org/)
- [`polars.DataFrame`](https://www.pola.rs/)
- [`polars.Series`](https://www.pola.rs/)
- [`torch.Tensor`](https://pytorch.org/)
- [`torch.nn.utils.rnn.PackedSequence`](https://pytorch.org/)
- [`xarray.DataArray`](https://docs.xarray.dev/en/stable/)
- [`xarray.Dataset`](https://docs.xarray.dev/en/stable/)
- [`xarray.Variable`](https://docs.xarray.dev/en/stable/)

Please check the [quickstart page](https://durandtibo.github.io/coola/quickstart) to learn more on
how to use `coola`.

## Documentation

- [latest (stable)](https://durandtibo.github.io/coola/): documentation from the latest stable
  release.
- [main (unstable)](https://durandtibo.github.io/coola/main/): documentation associated to the main
  branch of the repo. This documentation may contain a lot of work-in-progress/outdated/missing
  parts.

## Installation

We highly recommend installing
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
`coola` can be installed from pip using the following command:

```shell
pip install coola
```

To make the package as slim as possible, only the minimal packages required to use `coola` are
installed.
To include all the dependencies, you can use the following command:

```shell
pip install coola[all]
```

Please check the [get started page](https://durandtibo.github.io/coola/get_started) to see how to
install only some specific dependencies or other alternatives to install the library.
The following is the corresponding `coola` versions and tested dependencies.

| `coola` | `jax`<sup>*</sup> | `numpy`<sup>*</sup> | `pandas`<sup>*</sup> | `polars`<sup>*</sup> | `pyarrow`<sup>*</sup> | `torch`<sup>*</sup> | `xarray`<sup>*</sup> | `python`      |
|---------|-------------------|---------------------|----------------------|----------------------|-----------------------|---------------------|----------------------|---------------|
| `main`  | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14` |
| `0.8.5` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14` |
| `0.8.4` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.14` |
| `0.8.3` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.8.2` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.8.1` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.8.0` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.4` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      | `>=10.0,<18.0`        | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.3` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      |                       | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.2` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<2.0`      |                       | `>=1.11,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.1` | `>=0.4.1,<1.0`    | `>=1.21,<3.0`       | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.7.0` | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      |                       | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |

<sup>*</sup> indicates an optional dependency

<details>
    <summary>older versions</summary>

| `coola`  | `jax`<sup>*</sup> | `numpy`<sup>*</sup> | `pandas`<sup>*</sup> | `polars`<sup>*</sup> | `torch`<sup>*</sup> | `xarray`<sup>*</sup> | `python`      |
|----------|-------------------|---------------------|----------------------|----------------------|---------------------|----------------------|---------------|
| `0.6.2`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.6.1`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.6.0`  | `>=0.4.1,<1.0`    | `>=1.21,<2.0`       | `>=1.3,<3.0`         | `>=0.18.3,<1.0`      | `>=1.10,<3.0`       | `>=2023.1`           | `>=3.9,<3.13` |
| `0.5.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.4.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.3.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.3.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.2.2`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.2.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.2.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<1.0`      | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.1.2`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<0.21`     | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.1.1`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.13` |
| `0.1.0`  | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.12` |
| `0.0.26` | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     | `>=1.10,<2.2`       | `>=2023.1,<2023.13`  | `>=3.9,<3.12` |
| `0.0.25` | `>=0.4.1,<0.5`    | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     | `>=1.10,<2.2`       | `>=2023.4,<2023.11`  | `>=3.9,<3.12` |
| `0.0.24` | `>=0.3,<0.5`      | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     | `>=1.10,<2.2`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12` |
| `0.0.23` | `>=0.3,<0.5`      | `>=1.21,<1.27`      | `>=1.3,<2.2`         | `>=0.18.3,<0.20`     | `>=1.10,<2.1`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12` |
| `0.0.22` | `>=0.3,<0.5`      | `>=1.20,<1.26`      | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     | `>=1.10,<2.1`       | `>=2023.3,<2023.9`   | `>=3.9,<3.12` |
| `0.0.21` | `>=0.3,<0.5`      | `>=1.20,<1.26`      | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     | `>=1.10,<2.1`       | `>=2023.3,<2023.8`   | `>=3.9,<3.12` |
| `0.0.20` | `>=0.3,<0.5`      | `>=1.20,<1.26`      | `>=1.3,<2.1`         | `>=0.18.3,<0.19`     | `>=1.10,<2.1`       | `>=2023.3,<2023.8`   | `>=3.9`       |

</details>

## Contributing

Please check the instructions in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Suggestions and Communication

Everyone is welcome to contribute to the community.
If you have any questions or suggestions, you can
submit [Github Issues](https://github.com/durandtibo/coola/issues).
We will reply to you as soon as possible. Thank you very much.

## API stability

:warning: While `coola` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `coola` to a new version will possibly break any code that
was using the old version of `coola`.

## License

`coola` is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.

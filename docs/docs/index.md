# Home

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
    <a href="https://codecov.io/gh/durandtibo/coola">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/coola/branch/main/graph/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/coola/">
        <img alt="Documentation" src="https://github.com/durandtibo/coola/workflows/Documentation%20(stable)/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/coola/">
        <img alt="Documentation" src="https://github.com/durandtibo/coola/workflows/Documentation%20(unstable)/badge.svg">
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
is possible to extend it to [support other data structures](uguide/customization.md).

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

## API stability

:warning: While `coola` is in development stage, no API is guaranteed to be stable from one
release to the next. In fact, it is very likely that the API will change multiple times before a
stable 1.0.0 release. In practice, this means that upgrading `coola` to a new version will
possibly break any code that was using the old version of `coola`.

## License

`coola` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](https://github.com/durandtibo/coola/blob/main/LICENSE) file.

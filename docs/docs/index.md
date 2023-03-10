# Home

<p align="center">
    <a href="https://github.com/durandtibo/coola/actions">
        <img alt="CI" src="https://github.com/durandtibo/coola/workflows/CI/badge.svg?event=push&branch=main">
    </a>
    <a href="https://pypi.org/project/coola/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/coola">
    </a>
    <a href="https://pypi.org/project/coola/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/coola.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/coola">
    </a>
    <a href="https://codecov.io/gh/durandtibo/coola">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/coola/branch/main/graph/badge.svg">
    </a>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
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

`coola` is a light Python library that provides simple functions to check in a single line if two
complex/nested objects are equal or not.
`coola` was initially designed to work
with [PyTorch `Tensor`s](https://pytorch.org/docs/stable/tensors.html)
and [NumPy `ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html), but it
is possible to extend it to [support other data structures](customization.md).

## Motivation

Let's imagine you have the following dictionaries that contain both a PyTorch `Tensor` and a
NumPy `ndarray`.
You want to check if the two dictionaries are equal or not.
By default, Python does not provide an easy way to check if the two dictionaries are equal or not.
It is not possible to use the default equality operator `==` because it will raise an error.
The `coola` library was developed to fill this gap. `coola` provides a function `objects_are_equal`
that can indicate if two complex/nested objects are equal or not.

```python
import numpy
import torch

from coola import objects_are_equal

data1 = {'torch': torch.ones(2, 3), 'numpy': numpy.zeros((2, 3))}
data2 = {'torch': torch.zeros(2, 3), 'numpy': numpy.ones((2, 3))}

objects_are_equal(data1, data2)
```

`coola` also provides a function `objects_are_allclose` that can indicate if two complex/nested
objects are equal within a tolerance or not.

```python
from coola import objects_are_allclose

objects_are_allclose(data1, data2, atol=1e-6)
```

## API stability

:warning: While `coola` is in development stage, no API is guaranteed to be stable from one
release to the next. In fact, it is very likely that the API will change multiple times before a
stable 1.0.0 release. In practice, this means that upgrading `coola` to a new version will
possibly break any code that was using the old version of `coola`.

## License

`coola` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](https://github.com/durandtibo/coola/blob/main/LICENSE) file.

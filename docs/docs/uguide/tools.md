# Tools

:book: This page describes the `coola.tools` package, which provides general-purpose helpers that are
useful on their own and that `coola` itself does not need.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

`coola.tools` is separate from [`coola.utils`](utils.md), the foundation that the other `coola`
packages import. The modules are public and tested, but `coola` does not depend on them, so they
can be extracted to their own distribution later.

| Module | Content |
| --- | --- |
| `coola.tools.array` | Convert an object to a NumPy array |
| `coola.tools.tensor` | Convert an object to a PyTorch tensor, device checks |
| `coola.tools.batching` | Split an iterable in batches |
| `coola.tools.bloom_filter` | `BloomFilter`, an approximate set for duplicate detection |
| `coola.tools.date` | Today's date and conversion of dates |
| `coola.tools.deprecation` | Deprecate the positional use of options |
| `coola.tools.env_vars` | Read, check and temporarily set environment variables |
| `coola.tools.file_size` | The size of a file or of a directory |
| `coola.tools.git` | The branch and the commit of a git repository |
| `coola.tools.logging` | `CaptureHandler`, a logging handler that keeps the records |
| `coola.tools.lru` | `LRUCache`, a least-recently-used cache |
| `coola.tools.password` | Read a password from the terminal |
| `coola.tools.recursion` | Explain a `RecursionError` |
| `coola.tools.timing` | The `timeblock` context manager |

## Array Utilities (`coola.tools.array`)

The `coola.tools.array` module provides utilities for working with array-like objects across
different backends (NumPy, PyTorch). `to_array()` converts a sequence of numbers, a
`numpy.ndarray` or a `torch.Tensor` to a `numpy.ndarray`:

```pycon
>>> from coola.tools.array import to_array
>>> to_array([1, 2, 3])
array([1, 2, 3])

```

## Tensor Utilities (`coola.tools.tensor`)

The `coola.tools.tensor` module provides utilities for working with PyTorch tensors:

- `is_cuda_available()` - Check if CUDA is available with PyTorch
- `is_mps_available()` - Check if MPS (Apple Silicon) is available with PyTorch
- `get_available_devices()` - Get the PyTorch devices available on the machine
- `to_tensor()` - Convert a sequence of numbers, a `numpy.ndarray` or a `torch.Tensor` to a
  `torch.Tensor`

```pycon
>>> from coola.tools.tensor import is_cuda_available, is_mps_available
>>> is_cuda_available()  # doctest: +SKIP
False
>>> is_mps_available()  # doctest: +SKIP
False

```

## Bloom Filter (`coola.tools.bloom_filter`)

The `coola.tools.bloom_filter` module provides `BloomFilter`, a fixed-memory approximate set
used for duplicate detection. It never reports a false negative, but it can report a false
positive at a configurable rate. It is a standalone utility and is not used by the other
`coola` packages.

```pycon
>>> from coola.tools.bloom_filter import BloomFilter
>>> bf = BloomFilter(expected_items=1000, fp_rate=0.01)
>>> bf.add_and_check(b"hello")
False
>>> bf.add_and_check(b"hello")
True

```

## Environment Variable Utilities (`coola.tools.env_vars`)

The `coola.tools.env_vars` module provides utilities for working with environment variables:

- `get_required_env_var()` - Get an environment variable and raise a `ValueError` if it is not set
  or is empty
- `check_env_vars()` - Indicate which environment variables are defined, and optionally raise or log
  at a given level when some are missing
- `temp_env_vars()` - Context manager that temporarily sets environment variables and restores them
  on exit

```pycon
>>> from coola.tools.env_vars import check_env_vars, get_required_env_var, temp_env_vars
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

## Date Utilities (`coola.tools.date`)

The `coola.tools.date` module provides `get_today_date()` (today's date in a timezone, as a
string), `to_date()` and `to_datetime()` (convert a string to a `date` or a `datetime`):

```pycon
>>> from coola.tools.date import to_date, to_datetime
>>> to_date("2024-01-31")
datetime.date(2024, 1, 31)
>>> to_datetime("2024-01-31T10:00:00")
datetime.datetime(2024, 1, 31, 10, 0)

```

## See Also

- [Utility Functions](utils.md)
- [API reference](../refs/tools.md)

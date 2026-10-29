# Reducing Sequences

:book: This page describes the `coola.reducer` package, which provides a unified interface for
computing statistics and reductions on sequences of numeric values using different backends.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.reducer` package provides a consistent interface for computing common statistical
operations on sequences of numbers. It offers:

1. **Multiple backends** - Use native Python, NumPy, or PyTorch for computations
2. **Consistent API** - Same interface regardless of backend
3. **Automatic backend selection** - Automatically choose the best available backend
4. **Common operations** - min, max, mean, median, quantile, std, and sort

This is particularly useful when you want to write backend-agnostic code or when you need to
compute statistics on data but don't want to depend on a specific library.

## Properties

A reducer turns a one-dimensional sequence of real numbers into a statistic (`max`, `min`, `mean`,
`median`, `std`, `quantile`) or into a sorted list. It exists so that code written once runs with
whatever numeric library the user has, and gives the same answer.

| Property                    | Statement                                                                                                                                                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Same interface              | `NativeReducer`, `NumpyReducer` and `TorchReducer` have the same methods, the same errors and the same result types; the backend can be swapped. A backend only computes.                                                                                              |
| One input rule              | The input is a sequence of real numbers (`list`, `tuple`, `range`), or a one-dimensional array / tensor for the backend of the library. An iterator, a set, a string, a 0-d or n-d array is rejected with an error that says what to pass.                             |
| Python results              | The result is a Python `int` or `float` (a `list` of them for `quantile` and `sort`), never a NumPy or torch scalar. `max`, `min` and `sort` keep the type of the elements; `mean`, `std` and `quantile` are `float`; the median of an odd number of ints is an `int`. |
| Explicit empty case         | `max`, `min`, `mean`, `median`, `std` and `quantile` raise `EmptySequenceError` (a `ValueError`) on an empty input. `sort([])` is `[]`. `std` of one value is `NaN`.                                                                                                   |
| Valid options first         | `quantiles` outside `[0, 1]` (or `NaN`) raise `ValueError` before the input is read, with the same type and message in every backend.                                                                                                                                  |
| NaN                         | If a value is `NaN`, `max`, `min`, `mean`, `median`, `std` and every quantile are `NaN`, whatever the position of the `NaN`. `sort` puts the `NaN` last (first if `descending`). `std` is `NaN` if a value is infinite.                                                |
| No silent overflow          | The median of finite values is finite (the two middle values are halved before the mean if their sum overflows).                                                                                                                                                       |
| Unsupported values are loud | A value that the backend cannot represent (an integer beyond 64 bits for NumPy / torch, a complex number, a `Decimal`) raises `TypeError` that names the backend and, when it exists, the backend that supports it (`NativeReducer`).                                  |
| Pure                        | The input is never modified or consumed. A reducer is stateless, so one instance can be shared by threads and calls.                                                                                                                                                   |
| Lazy backends               | `import coola.reducer` imports no optional library. `NumpyReducer()` / `TorchReducer()` raise `MissingPackageError` at construction if the library is missing.                                                                                                         |
| Automatic backend           | `auto_reducer()` returns the fastest backend that is installed for plain Python sequences: NumPy, then native. PyTorch is never chosen implicitly (it is for tensors that live on a device, which the user passes to `TorchReducer`).                                  |
| Extensible                  | A backend is added by subclassing `BaseReducer` and implementing `_prepare` and six computations. It cannot forget the empty, iterator, quantile or shape checks, because it does not write them.                                                                      |

Not guaranteed:

- **Bit-identical floating point** across backends: the mean of `[0.1, 0.2, 0.3]` is `0.2` for
  `NativeReducer` and `0.20000000000000004` for the other two.
- **Sums that exceed the range of a `float`**: `mean([1e308, 1e308])` is `1e308` for `NativeReducer`
  and `inf` for NumPy and torch.
- **Other number types** (`bool`, `Fraction`, `Decimal`): `NativeReducer` computes with what Python
  provides; NumPy and torch may reject them.
- **Speed of PyTorch** for a Python list: the values are converted to a tensor first, so
  `TorchReducer` is slower than `NumpyReducer` on lists.

## Available Reducers

### `NativeReducer`

Uses Python's standard library (`statistics` module) for computations. This reducer is always
available and has no dependencies.

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.mean([1, 2, 3, 4, 5])
3.0
>>> reducer.median([1, 2, 3, 4, 5])
3
>>> reducer.max([1, 2, 3, 4, 5])
5

```

### `NumpyReducer`

Uses NumPy for computations. This reducer is faster than `NativeReducer` for large datasets and
provides more numerical stability.

```pycon
>>> from coola.reducer import NumpyReducer
>>> reducer = NumpyReducer()
>>> reducer.mean([1, 2, 3, 4, 5])
3.0
>>> reducer.std([1, 2, 3, 4, 5])
1.58113...

```

**Note:** Requires NumPy to be installed.

### `TorchReducer`

Uses PyTorch for computations. This reducer can leverage GPU acceleration and is useful when
working with PyTorch tensors.

```pycon
>>> from coola.reducer import TorchReducer
>>> reducer = TorchReducer()
>>> reducer.mean([1, 2, 3, 4, 5])
3.0
>>> reducer.median([1, 2, 3, 4, 5])
3

```

**Note:** Requires PyTorch to be installed.

## Basic Usage

### Computing Statistics

All reducers provide the same methods for computing statistics:

#### Maximum Value

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.max([10, 5, 8, 3, 12])
12
>>> reducer.max([-5, -10, -3])
-3

```

#### Minimum Value

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.min([10, 5, 8, 3, 12])
3
>>> reducer.min([-5, -10, -3])
-10

```

#### Mean (Average)

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.mean([1, 2, 3, 4, 5])
3.0
>>> reducer.mean([10, 20, 30])
20.0

```

#### Median

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.median([1, 2, 3, 4, 5])
3
>>> reducer.median([1, 2, 3, 4])
2.5

```

#### Standard Deviation

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.std([1, 2, 3, 4, 5])
1.5811388300841898
>>> reducer.std([10, 10, 10])
0.0

```

#### Quantiles

Compute multiple quantiles at once:

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.quantile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.25, 0.5, 0.75])
[2.5, 5.0, 7.5]
>>> reducer.quantile([1, 2, 3, 4, 5], [0.0, 0.5, 1.0])
[1.0, 3.0, 5.0]

```

Quantiles are specified as values between 0 and 1, where:

- 0.0 = minimum value
- 0.5 = median
- 1.0 = maximum value

### Sorting Values

Sort sequences in ascending or descending order:

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.sort([5, 2, 8, 1, 9])
[1, 2, 5, 8, 9]
>>> reducer.sort([5, 2, 8, 1, 9], descending=True)
[9, 8, 5, 2, 1]

```

## NaN Values

The three reducers behave the same way if a sequence contains `NaN`, whatever its position:
`max`, `min`, `mean`, `median`, `std` and `quantile` return `NaN`, and `sort` puts the `NaN` values
last (first if `descending=True`). `std` also returns `NaN` if a value is infinite.

```pycon
>>> import math
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> math.isnan(reducer.max([float("nan"), 1.0, 2.0]))
True
>>> math.isnan(reducer.max([1.0, 2.0, float("nan")]))
True
>>> reducer.sort([2.0, 1.0])
[1.0, 2.0]

```

## Automatic Backend Selection

Use `auto_reducer()` to automatically select the best available reducer based on installed
packages:

```pycon
>>> from coola.reducer import auto_reducer
>>> reducer = auto_reducer()
>>> reducer.mean([1, 2, 3, 4, 5])
3.0

```

The selection follows this priority order:

1. **NumpyReducer** if NumPy is available
2. **NativeReducer** as fallback (always available)

`TorchReducer` is never chosen implicitly: on a plain list it is the slowest backend and importing
PyTorch is slow. Create a `TorchReducer` explicitly to reduce tensors that live on a device.

This is useful when writing library code that should work with whatever the user has installed:

```pycon
>>> from coola.reducer import auto_reducer
>>> def compute_statistics(data):
...     reducer = auto_reducer()
...     return {
...         "mean": reducer.mean(data),
...         "std": reducer.std(data),
...         "median": reducer.median(data),
...     }
>>> compute_statistics([1, 2, 3, 4, 5])
{'mean': 3.0, 'std': 1.58113..., 'median': 3}

```

## Error Handling

### Invalid Input

The input is a sequence of real numbers, or a one-dimensional array / tensor for the backend. An
iterator, a set, a string or a value that the backend cannot represent raises `TypeError`, and an
input that is not one-dimensional raises `ValueError`. The message says what to pass:

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.max({1, 2, 3})
Traceback (most recent call last):
    ...
TypeError: Cannot compute the maximum of a set, pass a sequence of numbers (e.g. a list)
>>> reducer.quantile([1, 2, 3], [2])
Traceback (most recent call last):
    ...
ValueError: quantiles must be in [0, 1], got 2

```

### Empty Sequences

`max`, `min`, `mean`, `median`, `std` and `quantile` raise `EmptySequenceError` when given an
empty sequence. `EmptySequenceError` is a `ValueError`:

```pycon
>>> from coola.reducer import NativeReducer, EmptySequenceError
>>> reducer = NativeReducer()
>>> try:
...     reducer.mean([])
... except EmptySequenceError as e:
...     print(f"Error: {e}")
Error: Cannot compute the mean because the sequence is empty

```

Unlike the statistics, sorting is well-defined for an empty sequence, so `sort()` returns
an empty list instead of raising an error:

```pycon
>>> from coola.reducer import NativeReducer
>>> reducer = NativeReducer()
>>> reducer.sort([])
[]

```

## Common Use Cases

### Data Analysis

Compute summary statistics for data analysis:

```pycon
>>> from coola.reducer import auto_reducer
>>> reducer = auto_reducer()
>>> temperatures = [72, 68, 75, 71, 73, 69, 74]
>>> print(f"Mean: {reducer.mean(temperatures):.1f}°F")
Mean: 71.7°F
>>> print(f"Median: {reducer.median(temperatures):.1f}°F")
Median: 72.0°F
>>> print(f"Std Dev: {reducer.std(temperatures):.1f}°F")
Std Dev: 2.6°F
>>> print(f"Range: {reducer.min(temperatures)}-{reducer.max(temperatures)}°F")
Range: 68-75°F

```

### Performance Metrics

Analyze performance metrics:

```pycon
>>> from coola.reducer import auto_reducer
>>> reducer = auto_reducer()
>>> response_times = [120, 145, 132, 128, 156, 141, 139]
>>> quartiles = reducer.quantile(response_times, [0.25, 0.5, 0.75])
>>> print(f"25th percentile: {quartiles[0]:.1f}ms")
25th percentile: 130.0ms
>>> print(f"50th percentile (median): {quartiles[1]:.1f}ms")
50th percentile (median): 139.0ms
>>> print(f"75th percentile: {quartiles[2]:.1f}ms")
75th percentile: 143.0ms

```

### Ranking and Sorting

Sort values to find top performers:

```pycon
>>> from coola.reducer import auto_reducer
>>> reducer = auto_reducer()
>>> scores = [85, 92, 78, 95, 88, 91, 87]
>>> top_3 = reducer.sort(scores, descending=True)[:3]
>>> print(f"Top 3 scores: {top_3}")
Top 3 scores: [95, 92, 91]

```

## Design Principles

The `coola.reducer` package follows these design principles:

1. **Consistent interface**: All reducers implement the same `BaseReducer` interface
2. **Backend flexibility**: Choose the backend that fits your needs (native, NumPy, PyTorch)
3. **Automatic selection**: Use `auto_reducer()` to automatically select the best backend
4. **Clear error handling**: Explicit `EmptySequenceError` for operations on empty sequences

## What `coola.reducer` does not do

- **More statistics.** The six methods and `sort` are the contract (no `variance`, `mode`, `sum`,
  `argmax`, `percentile` or `histogram`). `std` is the sample standard deviation (`ddof=1`) in the
  three backends.
- **Axis or multi-dimensional reductions, weights and `skipna`.** A reducer reduces one sequence.
- **A registry of reducers.** There are three classes and the backend is chosen by the instance that
  you create or by `auto_reducer()`. `auto_reducer()` does not look at the values: for a tensor
  that lives on a device, build `TorchReducer()`. For a CPU tensor without gradient, `NumpyReducer`
  also works.
- **Lazy or streaming input.** An iterator is rejected on purpose, because the reducers go over the
  values several times: convert it to a list.
- **Precision beyond `float64`.** `NativeReducer` happens to work with `Decimal` and `Fraction`,
  but this is not a property.
- **Equality of reducers.** Two reducers of the same class are not equal (`==` is the identity).
  `NativeReducer.mean` uses `statistics.fmean`, which is fast and can differ from the exact mean
  in the last bit.

## Advanced Usage

### Creating a Custom Reducer

You can create a custom reducer by subclassing `BaseReducer`. The public methods (`max`, `mean`,
`median`, `min`, `quantile`, `std` and `sort`) are implemented by the base class: they check the
options and the empty case, then call the private computations. A subclass implements `_max`,
`_mean`, `_median`, `_min`, `_quantile`, `_std` and `_sort`, and optionally `_prepare` to convert
the input. The computations receive non-empty values (`_sort` can receive an empty input) and
quantiles in `[0, 1]`:

```pycon
>>> from coola.reducer import BaseReducer
>>> from collections.abc import Sequence
>>> class CustomReducer(BaseReducer[Sequence[float]]):
...     def _max(self, values: Sequence[float]) -> int | float:
...         return max(values)
...
...     def _mean(self, values: Sequence[float]) -> float:
...         return sum(values) / len(values)
...
...     def _median(self, values: Sequence[float]) -> int | float:
...         sorted_vals = sorted(values)
...         n = len(sorted_vals)
...         mid = n // 2
...         if n % 2 == 0:
...             return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
...         return sorted_vals[mid]
...
...     def _min(self, values: Sequence[float]) -> int | float:
...         return min(values)
...
...     def _quantile(self, values: Sequence[float], quantiles: Sequence[float]) -> list[float]:
...         # Custom quantile implementation
...         sorted_vals = sorted(values)
...         n = len(sorted_vals)
...         return [sorted_vals[int(q * (n - 1))] for q in quantiles]
...
...     def _std(self, values: Sequence[float]) -> float:
...         m = self._mean(values)
...         return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5
...
...     def _sort(self, values: Sequence[float], *, descending: bool) -> list[int | float]:
...         return sorted(values, reverse=descending)
>>> reducer = CustomReducer()
>>> reducer.mean([1, 2, 3, 4, 5])
3.0
>>> reducer.sort([])
[]

```

## See Also

- [`coola.utils`](utils.md): For utility functions including import helpers
- [`coola.summary`](summary.md): For summarizing data structures

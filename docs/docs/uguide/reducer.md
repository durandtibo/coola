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
1. **TorchReducer** if PyTorch is available
2. **NumpyReducer** if NumPy is available
3. **NativeReducer** as fallback (always available)

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
...
>>> compute_statistics([1, 2, 3, 4, 5])
{'mean': 3.0, 'std': 1.58113..., 'median': 3}

```

## Error Handling

### Empty Sequences

Most operations raise `EmptySequenceError` when given an empty sequence:

```pycon

>>> from coola.reducer import NativeReducer, EmptySequenceError
>>> reducer = NativeReducer()
>>> try:
...     reducer.mean([])
... except EmptySequenceError as e:
...     print(f"Error: {e}")
...
Error: Cannot compute the mean because the sequence is empty

```

Operations that raise `EmptySequenceError` on empty sequences:
- `max()`
- `min()`
- `mean()`
- `median()`
- `std()`
- `quantile()`

The `sort()` method does not raise an error for empty sequences; it returns an empty list:

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
5. **Type safety**: Generic types ensure type correctness

## Advanced Usage

### Creating a Custom Reducer

You can create custom reducers by extending `BaseReducer` or `BaseBasicReducer`:

```pycon

>>> from coola.reducer import BaseBasicReducer
>>> from collections.abc import Sequence
>>> from typing import TypeVar
>>> T = TypeVar("T", bound=Sequence[float])
>>> class CustomReducer(BaseBasicReducer[T]):
...     def _is_empty(self, values: T) -> bool:
...         return len(values) == 0
...     def _max(self, values: T) -> int | float:
...         return max(values)
...     def _mean(self, values: T) -> float:
...         return sum(values) / len(values)
...     def _median(self, values: T) -> int | float:
...         sorted_vals = sorted(values)
...         n = len(sorted_vals)
...         mid = n // 2
...         if n % 2 == 0:
...             return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
...         return sorted_vals[mid]
...     def _min(self, values: T) -> int | float:
...         return min(values)
...     def _quantile(self, values: T, quantiles: Sequence[float]) -> list[float]:
...         # Custom quantile implementation
...         sorted_vals = sorted(values)
...         n = len(sorted_vals)
...         return [sorted_vals[int(q * (n - 1))] for q in quantiles]
...     def _std(self, values: T) -> float:
...         m = self._mean(values)
...         return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5
...     def sort(self, values: T, descending: bool = False) -> list[int | float]:
...         return sorted(values, reverse=descending)
...
>>> reducer = CustomReducer()
>>> reducer.mean([1, 2, 3, 4, 5])
3.0

```

## See Also

- [`coola.utils`](utils.md): For utility functions including import helpers
- [`coola.summary`](summary.md): For summarizing data structures

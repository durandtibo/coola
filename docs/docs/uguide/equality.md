# Equality Comparison

:book: This page describes the `coola.equality` package, which provides a powerful and flexible
system for comparing objects of different types recursively.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.equality` package provides two main functions for checking equality between objects:

- `objects_are_equal()`: Check if two objects are equal
- `objects_are_allclose()`: Check if two objects are equal within a tolerance

These functions work with a wide variety of types, including:

- Python built-in types (int, float, str, list, dict, tuple, etc.)
- NumPy arrays
- PyTorch tensors
- pandas DataFrames and Series
- polars DataFrames and Series
- JAX arrays
- xarray DataArrays and Datasets
- PyArrow arrays and tables

The comparison is recursive, so nested structures like lists of dictionaries containing arrays are
handled correctly.

## Basic Usage

### Checking Exact Equality

Use `objects_are_equal()` to check if two objects are exactly equal:

```pycon

>>> from coola.equality import objects_are_equal
>>> objects_are_equal([1, 2, 3], [1, 2, 3])
True
>>> objects_are_equal([1, 2, 3], [1, 2, 4])
False

```

This works with nested structures:

```pycon

>>> from coola.equality import objects_are_equal
>>> objects_are_equal(
...     {"a": [1, 2], "b": {"x": 3}},
...     {"a": [1, 2], "b": {"x": 3}},
... )
True
>>> objects_are_equal(
...     {"a": [1, 2], "b": {"x": 3}},
...     {"a": [1, 2], "b": {"x": 4}},
... )
False

```

### Checking Equality with Tolerance

Use `objects_are_allclose()` to check if two objects are equal within a tolerance:

```pycon

>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(1.0, 1.0000001, atol=1e-6)
True
>>> objects_are_allclose(1.0, 1.1, atol=0.1)
True
>>> objects_are_allclose(1.0, 1.1, rtol=0.1)
True

```

The tolerance parameters are:

- `atol`: Absolute tolerance (default: 1e-8)
- `rtol`: Relative tolerance (default: 1e-5)

Two values are considered equal if: `|actual - expected| <= atol + rtol * |expected|`

This is particularly useful for comparing floating-point numbers and arrays:

```pycon

>>> import numpy as np
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(
...     np.array([1.0, 2.0, 3.0]),
...     np.array([1.0, 2.0, 3.0001]),
...     atol=1e-3,
... )
True

```

### Handling NaN Values

By default, NaN values are not considered equal to each other (following standard IEEE 754
behavior):

```pycon

>>> import numpy as np
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(np.array([1.0, float("nan")]), np.array([1.0, float("nan")]))
False

```

You can change this behavior with the `equal_nan` parameter:

```pycon

>>> import numpy as np
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(
...     np.array([1.0, float("nan")]),
...     np.array([1.0, float("nan")]),
...     equal_nan=True,
... )
True

```

The same parameter works with `objects_are_allclose()`:

```pycon

>>> import numpy as np
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(
...     np.array([1.0, float("nan")]),
...     np.array([1.0, float("nan")]),
...     equal_nan=True,
... )
True

```

### Showing Differences

When objects are not equal, you can display the differences using the `show_difference` parameter:

```pycon

>>> from coola.equality import objects_are_equal
>>> objects_are_equal(
...     {"a": [1, 2, 3], "b": 4},
...     {"a": [1, 2, 5], "b": 4},
...     show_difference=True,
... )
False

```

This will print detailed information about where the objects differ, which is very helpful for
debugging.

## Advanced Usage

### Controlling Recursion Depth

For deeply nested structures, you can control the maximum recursion depth to prevent stack
overflow. The `max_depth` parameter sets the maximum nesting level that will be compared:

```pycon

>>> from coola.equality import objects_are_equal
>>> # Simple nested structure
>>> nested1 = {"a": {"b": {"c": 1}}}
>>> nested2 = {"a": {"b": {"c": 1}}}
>>> objects_are_equal(nested1, nested2, max_depth=10)
True
>>> # Comparing lists of nested dicts
>>> data1 = [{"items": [{"value": i} for i in range(5)]}]
>>> data2 = [{"items": [{"value": i} for i in range(5)]}]
>>> objects_are_equal(data1, data2, max_depth=100)
True

```

The default `max_depth` is 1000, which should be sufficient for most use cases. If you have
extremely deeply nested structures (e.g., recursive data structures built programmatically), you
can increase this limit to avoid `RecursionError`.

### Using Custom Registries

The equality system uses a registry to dispatch to the appropriate comparison logic based on type.
You can provide a custom registry if needed:

```pycon

>>> from coola.equality import objects_are_equal
>>> from coola.equality.tester import get_default_registry
>>> registry = get_default_registry()
>>> objects_are_equal([1, 2, 3], [1, 2, 3], registry=registry)
True

```

For most use cases, you don't need to worry about the registry - the default registry handles all
supported types automatically.

## Type-Specific Behavior

### Python Built-in Types

For Python built-in types like `int`, `float`, `str`, `list`, `dict`, and `tuple`, the comparison
follows standard Python equality semantics with recursive comparison for collections.

See the [Supported Types](equality_types.md) page for detailed rules.

### NumPy Arrays

NumPy arrays are compared element-wise, checking:

- Same data type (`dtype`)
- Same shape
- Same values

```pycon

>>> import numpy as np
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
True
>>> objects_are_equal(
...     np.array([1, 2, 3], dtype=int),
...     np.array([1, 2, 3], dtype=float),
... )
False

```

### PyTorch Tensors

PyTorch tensors are compared checking:

- Same data type (`dtype`)
- Same device (CPU/GPU)
- Same shape
- Same values

```pycon

>>> import torch
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
True
>>> objects_are_equal(
...     torch.tensor([1, 2, 3], dtype=torch.float32),
...     torch.tensor([1, 2, 3], dtype=torch.int64),
... )
False

```

### pandas DataFrames

pandas DataFrames and Series are compared using pandas' built-in comparison methods.

```pycon

>>> import pandas as pd
>>> from coola.equality import objects_are_equal
>>> df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
>>> df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
>>> objects_are_equal(df1, df2)
True

```

## Common Use Cases

### Testing

The equality functions are particularly useful in unit tests:

```pycon

>>> from coola.equality import objects_are_equal
>>> def process_data(data):
...     # Some data processing
...     return {"result": [x * 2 for x in data]}
...
>>> result = process_data([1, 2, 3])
>>> expected = {"result": [2, 4, 6]}
>>> assert objects_are_equal(result, expected)

```

### Comparing Model Outputs

When working with machine learning models, you often need to compare outputs:

```pycon

>>> import torch
>>> from coola.equality import objects_are_allclose
>>> output1 = {
...     "predictions": torch.tensor([0.1, 0.9]),
...     "scores": torch.tensor([0.5, 0.8]),
... }
>>> output2 = {
...     "predictions": torch.tensor([0.1, 0.9]),
...     "scores": torch.tensor([0.5, 0.8]),
... }
>>> objects_are_allclose(output1, output2, atol=1e-6)
True

```

### Data Validation

Verify that data transformations preserve expected properties:

```pycon

>>> import numpy as np
>>> from coola.equality import objects_are_allclose
>>> original = np.array([1.0, 2.0, 3.0])
>>> transformed = np.array([1.0, 2.0, 3.0]) + 1e-10  # Small numerical error
>>> objects_are_allclose(original, transformed, atol=1e-8)
True

```

## Design Principles

The `coola.equality` package is designed around several key principles:

1. **Type-aware**: Different types are compared using appropriate logic (e.g., NumPy arrays use
   array comparison, not object comparison)

2. **Recursive**: Nested structures are compared recursively, so you can compare complex data
   structures

3. **Extensible**: The registry system allows adding support for new types without modifying the
   core library

4. **Configurable**: Parameters like `atol`, `rtol`, `equal_nan`, and `show_difference` give you
   fine-grained control

5. **Consistent**: The same comparison logic is used whether objects are at the top level or nested
   deep in a structure

## See Also

- [Supported Types](equality_types.md): Detailed rules for each supported type
- [`coola.equality.handler`](equality_handler.md): Learn about the handler system for implementing custom comparisons
- [`coola.equality.tester`](equality_tester.md): Learn about the tester registry system

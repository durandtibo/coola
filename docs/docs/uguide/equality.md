# Equality Comparison

:book: This page describes the `coola.equality` package, which provides a powerful and flexible
system for comparing objects of different types recursively. This page explains how to use the two
main functions of `coola`: `objects_are_equal` and `objects_are_allclose`. These functions can be
used to check if two complex/nested objects are equal or not.
The motivation of the library is explained [here](../index.md#motivation).

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).
It is highly recommended to know a bit of [NumPy](https://numpy.org/doc/stable/user/quickstart.html)
or [PyTorch](https://pytorch.org/tutorials/).

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

## Checking Exact Equality

### First Example

The following example shows how to use the `objects_are_equal` function.
The objects to compare are dictionaries containing a PyTorch `Tensor` and a NumPy `ndarray`.

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> data3 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> objects_are_equal(data1, data2)
False
>>> objects_are_equal(data1, data3)
True

```

In one line, it is possible to check two complex/nested objects are equal or not.
Unlike the native python equality operator `==`, the `objects_are_equal` function can check if two
dictionaries containing PyTorch `Tensor`s and NumPy `ndarray`s are equal or not.

### Finding a Difference

When the objects are complex or nested, it is not obvious to know which elements are different.
This function has an argument `show_difference` which shows the first difference found between the
two objects. For example if you add `show_difference=True` when you compare the `data1`
and `data2`, you will see at least one element that is different:

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> objects_are_equal(data1, data2, show_difference=True)
False

```

*Output*:

```textmate
torch.Tensors are different:
  actual   : tensor([[1., 1., 1.],
        [1., 1., 1.]])
  expected : tensor([[0., 0., 0.],
        [0., 0., 0.]])
mappings have different values:
  different value for key 'torch':
    actual   : tensor([[1., 1., 1.],
        [1., 1., 1.]])
    expected : tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

The output automatically appears when `show_difference=True` is used - no manual logging
configuration is needed. The output shows a clear, structured view of the difference
between `data1` and `data2`: the PyTorch `Tensor`s in key `'torch'` of the input dictionaries.
The output shows the element that fails the check first, and then shows the parent element, so
it is easy to know where the identified difference is located.

Note that it only shows the first difference, not all the differences. Two objects are different if
any of these elements are different. In the previous example, only the difference for key `'torch'`
is shown in the output.
No output is shown if the two objects are equal and `show_difference=True`.

**Key improvements in difference reporting:**

- **Automatic output**: No need to configure logging manually - output is automatically displayed
- **Structured formatting**: Clear indentation and organization
- **Specific information**: Shows exactly which index, key, or attribute differs
- **User-friendly**: Easy to read and understand at a glance

For example, with sequences, the output shows the specific index where values differ:

```pycon

>>> from coola.equality import objects_are_equal
>>> objects_are_equal([1, 2, 3, 4], [1, 2, 5, 4], show_difference=True)
False

```

*Output*:

```textmate
numbers are different:
  actual   : 3
  expected : 5
sequences have different values:
  different value at index 2:
    actual   : 3
    expected : 5
```

For mappings with different keys:

```pycon

>>> from coola.equality import objects_are_equal
>>> objects_are_equal({'a': 1, 'b': 2}, {'a': 1, 'c': 3}, show_difference=True)
False

```

*Output*:

```textmate
mappings have different keys:
  missing keys    : ['b']
  additional keys : ['c']
```

### More Examples

The previous examples use dictionary, but it is possible to use other types like list or tuple

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_equal
>>> data1 = [torch.ones(2, 3), numpy.zeros((2, 3))]
>>> data2 = [torch.zeros(2, 3), numpy.ones((2, 3))]
>>> data3 = (torch.ones(2, 3), numpy.zeros((2, 3)))
>>> objects_are_equal(data1, data2)
False
>>> objects_are_equal(data1, data3)
False

```

It is also possible to test more complex objects

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_equal
>>> data1 = {
...     "list": [torch.ones(2, 3), numpy.zeros((2, 3))],
...     "dict": {"torch": torch.arange(5), "str": "abc"},
...     "int": 1,
... }
>>> data2 = {
...     "list": [torch.ones(2, 3), numpy.zeros((2, 3))],
...     "dict": {"torch": torch.arange(5), "str": "abcd"},
...     "int": 1,
... }
>>> objects_are_equal(data1, data2)
False

```

Feel free to try any complex nested structure that you want. You can find the currently supported
types [here](equality_types.md#equal).

### Strict Type Checking

:warning: Unlike the native python equality operator `==`, the `objects_are_equal` function requires
two
objects to be of the same type to be equal.
For example, `1` (integer) is considered different from `1.0` (float) or `True` (boolean) which is
different behavior that the native python equality operator `==`. You can take a look to the
following example to see some differences.

```pycon
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(1, 1)
True
>>> objects_are_equal(1, 1.0)
False
>>> objects_are_equal(1, True)
False
>>> 1 == 1
True
>>> 1 == 1.0
True
>>> 1 == True
True

```

Similarly, the `objects_are_equal` function considers a `dict` and `collections.OrderedDict` as
different objects even if they have the same keys and values.

```pycon
>>> from collections import OrderedDict
>>> from coola.equality import objects_are_equal
>>> objects_are_equal({"key1": 1, "key2": "abc"}, OrderedDict({"key1": 1, "key2": "abc"}))
False
>>> {"key1": 1, "key2": "abc"} == OrderedDict({"key1": 1, "key2": "abc"})
True

```

## Checking Equality with Tolerance

### First Example

`coola` provides a function `objects_are_allclose` that can indicate if two complex/nested objects
are equal within a tolerance or not.
Due to numerical precision, it happens quite often that two numbers are not equal but the error is
very tiny (`1.0` and `1.000000001`).
The tolerance is mostly useful for numerical values.
For a lot of types like string, the `objects_are_allclose` function behaves like
the `objects_are_equal` function.

The following example shows how to use the `objects_are_allclose` function.
The objects to compare are dictionaries containing a PyTorch Tensor and a NumPy ndarray.

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_allclose, objects_are_equal
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> data3 = {"torch": torch.ones(2, 3) + 1e-9, "numpy": numpy.zeros((2, 3)) - 1e-9}
>>> objects_are_allclose(data1, data2)
False
>>> objects_are_allclose(data1, data3)
True
>>> objects_are_equal(data1, data3)
False

```

The difference between `data1` and `data2` is large so `objects_are_allclose` returns false
like `objects_are_equal`. The difference between `data1` and `data3` is tiny
so `objects_are_allclose` returns true, whereas `objects_are_equal` returns false.

### Tolerance

It is possible to control the tolerance with the arguments `atol` and `rtol`. `atol` controls the
absolute tolerance and `rtol` controls the relative tolerance.

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_allclose
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.ones(2, 3) + 1e-4, "numpy": numpy.zeros((2, 3)) - 1e-4}
>>> objects_are_allclose(data1, data2)
False
>>> objects_are_allclose(data1, data2, atol=1e-3)
True

```

`objects_are_equal` and `objects_are_allclose` are very similar and should behave the same
when `atol=0.0` and `rtol=0.0`.

The tolerance parameters are:

- `atol`: Absolute tolerance (default: 1e-8)
- `rtol`: Relative tolerance (default: 1e-5)

Two values are considered equal if: `|actual - expected| <= atol + rtol * |expected|`

### Finding a Difference

Like `objects_are_equal`, the `objects_are_allclose` function has an argument `show_difference`
which shows the first difference found between the two objects.

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_allclose
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.ones(2, 3) + 1e-4, "numpy": numpy.zeros((2, 3)) - 1e-4}
>>> objects_are_allclose(data1, data2, show_difference=True)
False

```

*Output*:

```textmate
torch.Tensors are different:
  actual   : tensor([[1., 1., 1.],
        [1., 1., 1.]])
  expected : tensor([[1.0001, 1.0001, 1.0001],
        [1.0001, 1.0001, 1.0001]])
mappings have different values:
  different value for key 'torch':
    actual   : tensor([[1., 1., 1.],
        [1., 1., 1.]])
    expected : tensor([[1.0001, 1.0001, 1.0001],
        [1.0001, 1.0001, 1.0001]])
```

The difference output uses the same improved formatting as `objects_are_equal`, with
clear indentation and specific information about what differs and where.

### More Examples

Like the `objects_are_equal` function, the `objects_are_allclose` function can be used with
complex/nested objects.

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_allclose
>>> data1 = {
...     "list": [torch.ones(2, 3), numpy.zeros((2, 3))],
...     "dict": {"torch": torch.arange(5), "str": "abc"},
...     "int": 1,
... }
>>> data2 = {
...     "list": [torch.ones(2, 3), numpy.zeros((2, 3)) + 1e-9],
...     "dict": {"torch": torch.arange(5), "str": "abc"},
...     "int": 1,
... }
>>> objects_are_allclose(data1, data2)
True

```

`objects_are_allclose` supports a lot of types and nested structure.
Feel free to try any complex nested structure that you want. You can find the currently supported
types [here](equality_types.md#equal-within-a-tolerance-allclose).

### Handling NaN Values

By default, `NaN` is not considered close to any other value, including `NaN`.

```pycon
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(float("nan"), 0.0)
False
>>> objects_are_allclose(float("nan"), float("nan"))
False

```

By setting `equal_nan=True`, it is possible to change the above behavior and `NaN`s will be
considered equal.

```pycon
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(float("nan"), float("nan"), equal_nan=True)
True

```

In arrays or tensors, `NaN` are sometimes used to indicate some values are not valid.
However, it may be interesting to check if the non-`NaN` values are equal.
It is possible to use the `equal_nan=True` option to compare two tensors with `NaN` values.

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(
...     torch.tensor([0.0, 1.0, float("nan")]),
...     torch.tensor([0.0, 1.0, float("nan")]),
... )
False
>>> objects_are_allclose(
...     torch.tensor([0.0, 1.0, float("nan")]),
...     torch.tensor([0.0, 1.0, float("nan")]),
...     equal_nan=True,
... )
True
>>> objects_are_allclose(
...     numpy.array([0.0, 1.0, float("nan")]),
...     numpy.array([0.0, 1.0, float("nan")]),
... )
False
>>> objects_are_allclose(
...     numpy.array([0.0, 1.0, float("nan")]),
...     numpy.array([0.0, 1.0, float("nan")]),
...     equal_nan=True,
... )
True

```

The same parameter works with `objects_are_equal()`:

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

## Connection with Similar Tools

`coola` is not the first tool to provide functions to compare nested objects.
If you are a PyTorch user, you probably know
the [`torch.testing.assert_close`](https://pytorch.org/docs/stable/testing.html) function.
If you are a NumPy user, you probably know
the [
`numpy.testing.assert_equal`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_equal.html)
function.
However, most of these functions work in a fixed scope and are difficult to extend or customize.
On the opposite side, `coola` is flexible and easy to customize.

Let's take a look to `torch`.
[`torch.testing.assert_close`](https://pytorch.org/docs/stable/testing.html) allows to easily
compare `torch.Tensor`s objects:

```pycon
>>> import torch
>>> torch.testing.assert_close(torch.ones(2, 3), torch.ones(2, 3))

```

It can also be used on mappings or sequences:

```pycon
>>> import torch
>>> torch.testing.assert_close(
...     [torch.ones(2, 3), torch.zeros(3)],
...     [torch.ones(2, 3), torch.zeros(3)],
... )
>>> torch.testing.assert_close(
...     {"key1": torch.ones(2, 3), "key2": torch.zeros(3)},
...     {"key1": torch.ones(2, 3), "key2": torch.zeros(3)},
... )
>>> torch.testing.assert_close(
...     {
...         "key1": torch.ones(2, 3),
...         "key2": {"key3": torch.zeros(3), "key4": [torch.arange(5)]},
...     },
...     {
...         "key1": torch.ones(2, 3),
...         "key2": {"key3": torch.zeros(3), "key4": [torch.arange(5)]},
...     },
... )

```

It also works on tensor like objects like NumPy arrays:

```pycon
>>> import torch
>>> import numpy as np
>>> torch.testing.assert_close(
...     [torch.ones(2, 3), np.zeros(3)],
...     [torch.ones(2, 3), np.zeros(3)],
... )
>>> torch.testing.assert_close([torch.ones(2, 3), 42], [torch.ones(2, 3), 42])

```

However, it does not work if the data structure contains a string:

```pycon
>>> import torch
>>> torch.testing.assert_close(
...     {"key1": torch.ones(2, 3), "key2": torch.zeros(3), "key3": "abc"},
...     {"key1": torch.ones(2, 3), "key2": torch.zeros(3), "key3": "abc"},
... )
Traceback (most recent call last):
...
TypeError: No comparison pair was able to handle inputs of type <class 'str'> and <class 'str'>.
The failure occurred for item ['key3']

```

`coola` can compare these objects:

```pycon
>>> import torch
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(
...     {"key1": torch.ones(2, 3), "key2": torch.zeros(3), "key3": "abc"},
...     {"key1": torch.ones(2, 3), "key2": torch.zeros(3), "key3": "abc"},
... )
True

```

Internally, [`torch.testing.assert_close`](https://pytorch.org/docs/stable/testing.html) tries to
convert some values to tensors to compare them, which can lead to surprising results like:

```pycon
>>> import torch
>>> torch.testing.assert_close((1, 2, 3), [1, 2, 3])

```

The inputs have different types: the left input is a tuple, whereas the right is a list.
`coola` has a strict type checking and will indicate the two inputs are different:

```pycon
>>> import torch
>>> from coola.equality import objects_are_equal
>>> objects_are_equal((1, 2, 3), [1, 2, 3])
False

```

[
`numpy.testing.assert_equal`](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_equal.html)
has different limitations.
For example, it can work with strings but can handle only simple sequence and mapping objects

```pycon
>>> import numpy as np
>>> from collections import deque
>>> np.testing.assert_equal(
...     {"key1": np.ones((2, 3)), "key2": np.zeros(3)},
...     {"key1": np.ones((2, 3)), "key2": np.zeros(3)},
... )
>>> np.testing.assert_equal(
...     {"key1": np.ones((2, 3)), "key2": np.zeros(3), "key3": "abc"},
...     {"key1": np.ones((2, 3)), "key2": np.zeros(3), "key3": "abc"},
... )
>>> np.testing.assert_equal(
...     deque([np.ones((2, 3)), np.zeros(3)]),
...     deque([np.ones((2, 3)), np.zeros(3)]),
... )
Traceback (most recent call last):
...
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

```

`coola` can compare these objects:

```pycon
>>> from coola.equality import objects_are_equal
>>> import numpy as np
>>> from collections import deque
>>> objects_are_equal(
...     {"key1": np.ones((2, 3)), "key2": np.zeros(3)},
...     {"key1": np.ones((2, 3)), "key2": np.zeros(3)},
... )
True
>>> objects_are_equal(
...     {"key1": np.ones((2, 3)), "key2": np.zeros(3), "key3": "abc"},
...     {"key1": np.ones((2, 3)), "key2": np.zeros(3), "key3": "abc"},
... )
True
>>> objects_are_equal(
...     deque([np.ones((2, 3)), np.zeros(3)]),
...     deque([np.ones((2, 3)), np.zeros(3)]),
... )
True

```

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

# Equality Comparison

:book: This page describes the `coola.equality` package, which provides a powerful and flexible
system for comparing objects of different types recursively. This page explains how to use the two
main functions of `coola`: `objects_are_equal` and `objects_are_allclose`. These functions can be
used to check if two complex/nested objects are equal or not.

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
- pydantic models

Nested structures like lists of dictionaries containing arrays are handled correctly. The comparison
uses an explicit stack, not the Python call stack, so the nesting depth cannot overflow the
interpreter.

## Guarantees

The comparison functions follow these rules for every type, in addition to the type-specific rules
described [below](#type-specific-behavior).

| Property                | Statement                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Strict type             | Two objects must have exactly the same type to be equal. `1`, `1.0` and `True` are pairwise different, and so are `[1]` and `(1,)`, `dict` and `OrderedDict`, `set` and `frozenset`, `str` and `bytes`. A subclass is different from its base class.                                                                                                                                                                                                                                                                                            |
| Recursive               | Containers are compared item by item, at any depth. Containers must have the same length (or the same keys).                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Order rules             | Sequences are order-sensitive (`[1, 2]` differs from `[2, 1]`). `dict` and other mappings ignore the order of the keys: `{"a": 1, "b": 2}` equals `{"b": 2, "a": 1}`. `OrderedDict` is the exception: the order of its keys is compared, like in Python. See [Ordered dictionaries](#ordered-dictionaries).                                                                                                                                                                                                                                     |
| Value semantics         | The values are compared with `==` unless a tester exists for the type. Arrays, tensors and data frames also compare dtype, shape and so on (see [Type-Specific Behavior](#type-specific-behavior)).                                                                                                                                                                                                                                                                                                                                             |
| NaN                     | `NaN` is not equal to `NaN`, unless `equal_nan=True` or both are the same object. See [Behavior by Type](#behavior-by-type-nan-and-tolerance).                                                                                                                                                                                                                                                                                                                                                                                                  |
| Signed zero             | `-0.0` is equal to `0.0`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Time zones              | Aware datetimes compare by the instant they represent: 12:00 UTC equals 13:00 UTC+1. A naive datetime (no time zone) never equals an aware one.                                                                                                                                                                                                                                                                                                                                                                                                 |
| Tolerance               | Only the `*_allclose` functions and `compare(..., rtol=..., atol=...)` use `rtol` and `atol`. `objects_are_equal` and `assert_objects_equal` are exact (no tolerance), `objects_are_allclose` and `assert_objects_allclose` default to `rtol=1e-5, atol=1e-8` (the NumPy defaults), and `compare` is exact unless `rtol` or `atol` is given. The tolerance applies only for numbers, arrays, tensors and data frames. Every other value is compared exactly, and the types must still match: `objects_are_allclose(1, 1.0, atol=1)` is `False`. |
| Unregistered types      | A type without a dedicated tester is compared with `==` after the type check, so your own classes work if they define `__eq__`. Otherwise two objects are equal only if they are the same object (a plain class, an exception, a `functools.partial`): register an `ObjectAttributesEqualityTester` to compare them by their attributes (see [Equality Testers](equality_tester.md)).                                                                                                                                                           |
| Reflexive               | `objects_are_equal(x, x)` is `True` for every `x`, and also for a copy of `x`. A distinct `NaN` is the exception unless `equal_nan=True` (see the NaN row); a null (`None`, `pd.NA`) is not an exception.                                                                                                                                                                                                                                                                                                                                       |
| Symmetric               | `objects_are_equal(a, b)` and `objects_are_equal(b, a)` give the same answer, and so do `objects_are_allclose(a, b)` and `objects_are_allclose(b, a)`. A user class with an asymmetric `__eq__` is the exception.                                                                                                                                                                                                                                                                                                                               |
| Not transitive          | With a tolerance, `a` close to `b` and `b` close to `c` does not imply that `a` is close to `c`. The exact comparison is transitive.                                                                                                                                                                                                                                                                                                                                                                                                            |
| Pure                    | The comparison does not modify its inputs, and gives the same answer every time.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| Failure modes           | A self-referencing container does not raise: a pair that is already being compared is considered equal. A structure nested deeper than `max_depth` raises a `MaxDepthExceededError` (see [Controlling Recursion Depth](#controlling-recursion-depth)). Unsupported types fall back to `==`: an `==` that raises propagates unchanged, and an `==` that returns an array (see [Types that work with caveats](#types-that-work-with-caveats)) raises a `TypeError`.                                                                               |
| Consistent with hashing | Objects that are equal also have the same [`hash_object`](hashing.md) hash. The reverse is not guaranteed: a `list` subclass and a `list`, or a NumPy `float64` and a `float`, share a hash but are not equal.                                                                                                                                                                                                                                                                                                                                  |

```pycon
>>> from collections import OrderedDict
>>> from coola.equality import objects_are_allclose, objects_are_equal
>>> objects_are_equal([1, 2], (1, 2))
False
>>> objects_are_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})
True
>>> objects_are_equal(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1))
False
>>> objects_are_equal(0.0, -0.0)
True
>>> objects_are_equal(float("nan"), float("nan"))
False
>>> objects_are_equal(float("nan"), float("nan"), equal_nan=True)
True
>>> objects_are_allclose([1.0, 2.0], [1.001, 2.001], atol=0.01)
True
>>> objects_are_allclose([1.0, 2.0], (1.001, 2.001), atol=0.01)
False

```

Time zones follow the same rule as Python's `==`:

```pycon
>>> from datetime import UTC, datetime, timedelta, timezone
>>> from coola.equality import objects_are_equal
>>> utc = datetime(2024, 1, 1, 12, tzinfo=UTC)
>>> plus_one = timezone(timedelta(hours=1))
>>> objects_are_equal(utc, datetime(2024, 1, 1, 13, tzinfo=plus_one))  # same instant
True
>>> objects_are_equal(utc, datetime(2024, 1, 1, 12, tzinfo=plus_one))  # 11:00 UTC
False
>>> objects_are_equal(utc, datetime(2024, 1, 1, 12))  # naive vs aware
False

```

### Ordered dictionaries

For a `dict`, the order of the keys is not meaningful, so it is ignored. An `OrderedDict` is
different: the order of the items is the reason to use it, and Python's `==` also compares the order
between two `OrderedDict`s. `objects_are_equal` and `objects_are_allclose` follow this design:

| Comparison                                  | Result  | Why                                          |
| ------------------------------------------- | ------- | -------------------------------------------- |
| `OrderedDict` vs `OrderedDict`, same order  | `True`  | Same type, same items, same order            |
| `OrderedDict` vs `OrderedDict`, other order | `False` | The order of the keys is compared            |
| `OrderedDict` vs `dict`                     | `False` | Different types (the strict type rule)       |
| `dict` vs `dict`, other order               | `True`  | The order of the keys of a `dict` is ignored |

```pycon
>>> from collections import OrderedDict
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(OrderedDict(a=1, b=2), OrderedDict(a=1, b=2))  # same order
True
>>> objects_are_equal(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1))  # different order
False
>>> objects_are_equal(OrderedDict(a=1, b=2), dict(a=1, b=2))  # different type
False
>>> objects_are_equal({"a": 1, "b": 2}, {"b": 2, "a": 1})  # plain dict: order ignored
True

```

The rule applies at any depth, so an `OrderedDict` nested in a list or in a `dict` is also checked.
The tolerance only applies to the values: `objects_are_allclose` still requires the same key order.
`compare` explains the difference:

```pycon
>>> from collections import OrderedDict
>>> from coola.equality import compare
>>> print(compare(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1)).format_message())
objects are not equal at actual
mappings have the same keys in a different order:
  actual   : ['a', 'b']
  expected : ['b', 'a']

```

Only `OrderedDict` (and its subclasses) is order-sensitive. For another mapping type whose order
matters, register an `OrderedMappingEqualityTester` (see
[`coola.equality.tester`](equality_tester.md)). The hash of `hash_object` follows the same design:
the hash of an `OrderedDict` depends on the order of its keys, and the hash of a `dict` does not.

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
`compare` returns a `ComparisonResult` that explains the first difference found between the two
objects: its location (`path`) and the reason. For example, you can see which element of `data1` and
`data2` is different:

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import compare
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.zeros(2, 3), "numpy": numpy.ones((2, 3))}
>>> result = compare(data1, data2)
>>> result.equal
False
>>> result.path
('torch', 0, 0)
>>> print(result.format_message())
objects are not equal at actual['torch'][0][0]
values are different:
  actual   : tensor(1.)
  expected : tensor(0.)

```

The result gives the location of the difference (for an array or a tensor, the index of the first
element that differs) and the two values that differ. Note that it only
shows the first difference, not all the differences. Two objects are different if any of these
elements are different. In the previous example, only the difference for key `'torch'` is explained.
`objects_are_equal` and `objects_are_allclose` do not build any explanation: they stop at the first
difference and only return a `bool`. `assert_objects_equal` and `assert_objects_allclose` raise an
`AssertionError` with the message of `compare`, which is convenient in tests.

For example, with sequences, the result shows the specific index where values differ:

```pycon
>>> from coola.equality import compare
>>> print(compare([1, 2, 3, 4], [1, 2, 5, 4]).format_message())
objects are not equal at actual[2]
numbers are different:
  actual   : 3
  expected : 5

```

For mappings with different keys:

```pycon
>>> from coola.equality import compare
>>> print(compare({"a": 1, "b": 2}, {"a": 1, "c": 3}).format_message())
objects are not equal at actual
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
types in the [Type-Specific Behavior](#type-specific-behavior) section.

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
different objects even if they have the same keys and values (see
[Ordered dictionaries](#ordered-dictionaries) for the order of the keys).

```pycon
>>> from collections import OrderedDict
>>> from coola.equality import objects_are_equal
>>> objects_are_equal({"key1": 1, "key2": "abc"}, OrderedDict({"key1": 1, "key2": "abc"}))
False
>>> {"key1": 1, "key2": "abc"} == OrderedDict({"key1": 1, "key2": "abc"})
True

```

The exact type is checked at every level of a nested structure, so the native `==` quirks do not
leak into containers: `[1]` and `[True]` are different, and so are `(1,)` and `(True,)` or
`{"a": 1}` and `{"a": True}`. This avoids surprising results such as `[1] == [True]` being `True`
with the native operator.

```pycon
>>> from coola.equality import objects_are_equal
>>> [1] == [True]
True
>>> objects_are_equal([1], [True])
False
>>> objects_are_equal({"a": 1}, {"a": True})
False

```

:warning: The keys of a `dict` are matched by their type and their value, so `{1: "a"}` and
`{True: "a"}` are different (unlike with `==`). The members of a `set` are matched by their
type and hash, so `{1}` and `{True}` are different (unlike with `==`).

```pycon
>>> from coola.equality import objects_are_equal
>>> {1: "a"} == {True: "a"}
True
>>> objects_are_equal({1: "a"}, {True: "a"})
False
>>> {1} == {True}
True
>>> objects_are_equal({1}, {True})
False

```

With `equal_nan=True`, two `NaN` keys (also inside a `tuple` key) are the same key, as the `NaN` members
of a `set` are:

```pycon
>>> from coola.equality import objects_are_equal
>>> objects_are_equal({float("nan"): 1}, {float("nan"): 1})
False
>>> objects_are_equal({float("nan"): 1}, {float("nan"): 1}, equal_nan=True)
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

`compare` takes the same `rtol` and `atol` keywords, but it is exact when they are not given,
whereas `objects_are_allclose` and `assert_objects_allclose` default to `rtol=1e-5, atol=1e-8`.
`compare(a, b, rtol=1e-5, atol=1e-8)` gives the same answer as `objects_are_allclose(a, b)`:

```pycon
>>> from coola.equality import compare
>>> bool(compare(1.0, 1.0 + 1e-9))
False
>>> bool(compare(1.0, 1.0 + 1e-9, rtol=1e-5, atol=1e-8))
True
>>> bool(compare(1.0, 1.05, atol=0.1))
True

```

Two numbers are considered equal if `|actual - expected| <= atol + rtol * max(|actual|, |expected|)`.
This is the rule of NumPy, PyTorch, JAX, PyArrow and xarray arrays, and it is also used for Python
numbers (`float`, `int`, `complex`), NumPy scalars, `Decimal`, `Fraction` and `timedelta`. It is
symmetric: `objects_are_allclose(a, b)` and `objects_are_allclose(b, a)` give the same result.
The numeric columns (integers and floats, also the nullable ones, and complex numbers for pandas)
and the duration columns of `pandas`, `polars` and `pyarrow` data frames, series and arrays follow
the same rule (`atol` of a duration is a number of seconds). The other columns (booleans, strings,
categories, instants, decimals, nested values and `object` columns) are compared exactly with the
`equals` of the library: the tolerance and `equal_nan` do not reach the values nested in a cell.
Whether a `NaN` nested in a cell equals a `NaN` is decided by the library: it does in `polars`, it
does not in `pyarrow`, and it does in `pandas` only if both cells hold the same `NaN` object.

The comparison within a tolerance is not transitive: `a` can be close to `b`, and `b` close to `c`,
while `a` is not close to `c`.

### Finding a Difference

Like `objects_are_equal`, `objects_are_allclose` only returns a `bool`. To find the first difference
between the two objects within a tolerance, use `compare` with the same `rtol` and `atol`
(`compare` is exact by default, `objects_are_allclose` uses `rtol=1e-5` and `atol=1e-8`):

```pycon
>>> import numpy
>>> import torch
>>> from coola.equality import compare
>>> data1 = {"torch": torch.ones(2, 3), "numpy": numpy.zeros((2, 3))}
>>> data2 = {"torch": torch.ones(2, 3) + 1e-4, "numpy": numpy.zeros((2, 3)) - 1e-4}
>>> result = compare(data1, data2, rtol=1e-5, atol=1e-8)
>>> result.equal
False
>>> result.path
('torch', 0, 0)

```

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
types in the [Type-Specific Behavior](#type-specific-behavior) section.

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

## Common options

All the entry points (`objects_are_equal`, `objects_are_allclose`, `compare`, `assert_objects_equal`
and `assert_objects_allclose`) share the same options, with the same meaning:

| Option      | Meaning                                                                                                                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `equal_nan` | Treat `NaN` as equal to `NaN`. See [Handling NaN Values](#handling-nan-values).                                                                                                                  |
| `max_depth` | The nesting limit (default 100): a deeper structure raises `MaxDepthExceededError`. See [Controlling Recursion Depth](#controlling-recursion-depth). It is not the display depth of `summarize`. |
| `registry`  | The registry that maps a type to its tester (default: the default registry). See [Using Custom Registries](#using-custom-registries).                                                            |

The `*_allclose` functions also take `rtol` and `atol`. `compare` takes `rtol` and `atol` too, and is
exact unless one of them is given. The assert functions take a `root` name (default `"actual"`) for
the message.

## Advanced Usage

### Controlling Recursion Depth

The `max_depth` option is the **nesting limit** of the comparison.

There are two special cases, and only the second raises:

| Case                                     | Result                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| Cycle (a container that contains itself) | No error: a pair that is already being compared is considered equal                  |
| Nesting deeper than `max_depth`          | `coola.equality.MaxDepthExceededError`, a plain `Exception` (not a `RecursionError`) |

`max_depth` must be a positive `int` (not a `bool`), otherwise a `ValueError` (not positive) or a
`TypeError` (not an `int`) is raised. `rtol` and `atol` must be non-negative finite real numbers
(not `bool`).

The comparison does not recurse: the engine keeps the child iterators on an explicit stack, so a
`RecursionError` cannot happen whatever the nesting (100 000 nested dictionaries compare under the
default interpreter limit). `max_depth` is therefore a policy, not a safety net: it bounds the work
and the memory on a pathological structure, for example an object that creates a new child each time
an attribute is read. The `max_depth` parameter sets the maximum nesting level that will be compared. It is a
**limit that raises** (`MaxDepthExceededError`) for a deeper structure, unlike the `max_depth` of
`summarize`, which is a **display depth** that truncates the output:

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

The default `max_depth` is 100, which is more than the nesting of usual data (JSON-like documents,
configurations, model outputs). Raise it if a legitimate structure is deeper.

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

:book: This section describes what types are currently supported and what rules are used to check if
two objects are equal or not.

### Supported Types

!!! note "These lists are not exhaustive"

    `coola` dispatches on the type of the first argument and, for a type without a specific
    tester, falls back to `actual == expected` (after checking that both objects have the same
    type). So many types that are not listed here also work, and some types listed as working
    well have limits. When a type matters for your use case, check it with a small example
    (see [Explaining a Difference](#explaining-a-difference)), or register your own tester (see
    [Using Custom Registries](#using-custom-registries)).

#### Types that work well

The exact equality, the tolerance (`atol`, `rtol`) and `equal_nan` are all handled for:

- Python numbers: `int`, `float`, `bool` and `complex`
- NumPy scalars (`numpy.floating`, `numpy.integer`, `numpy.complexfloating`, for example
  `numpy.float32`) and [`numpy.ndarray`](https://numpy.org/doc/stable/index.html) with a numeric
  dtype
- [`numpy.ma.MaskedArray`](https://numpy.org/doc/stable/reference/maskedarray.generic.html)
- [`jax.numpy.ndarray`](https://jax.readthedocs.io/en/latest/index.html)
- [`torch.Tensor`](https://pytorch.org/) and
  [`torch.nn.utils.rnn.PackedSequence`](https://pytorch.org/)
- [`xarray.DataArray`](https://docs.xarray.dev/en/stable/),
  [`xarray.Dataset`](https://docs.xarray.dev/en/stable/) and
  [`xarray.Variable`](https://docs.xarray.dev/en/stable/)
- The containers `list`, `tuple`, `collections.deque`, `dict`, `collections.OrderedDict` and
  the other `Sequence` and `Mapping` types: they are compared item by item, so the items can be
  of any supported type
- Dataclass instances: the fields with `compare=True` are compared with the registry (so a field
  can be a float, an array, another dataclass, and so on). A dataclass defined with `eq=False`
  keeps its own `__eq__`
- [`pydantic.BaseModel`](https://docs.pydantic.dev/) instances: the fields and the extra values
  are compared with the registry (so a field can be a float, an array, another model, and so on),
  and the private attributes are ignored

#### Types that work with caveats

- [`pandas.DataFrame`](https://pandas.pydata.org/) and
  [`pandas.Series`](https://pandas.pydata.org/): a `NaN` in a float or complex column is governed
  by `equal_nan`, whereas a null (`None`, `pd.NA`, `NaT`) in any other column is equal to a null,
  whatever `equal_nan`, so a copy of a frame is equal to the frame
- [`polars.DataFrame`](https://www.pola.rs/), [`polars.Series`](https://www.pola.rs/) and
  `polars.LazyFrame`: `null` is always equal to `null`, and a `LazyFrame` is collected, so the
  comparison runs the query
- [`pyarrow.Array`](https://arrow.apache.org/docs/python/generated/pyarrow.Array.html),
  `pyarrow.ChunkedArray`, `pyarrow.RecordBatch` and
  [`pyarrow.Table`](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html)
  (experimental): `equal_nan`, `atol` and `rtol` are honored for the integer and floating-point
  columns (this requires NumPy), and the other types are compared exactly
- `numpy.ndarray` with a non-numeric dtype: `str`, `bytes` and `object` arrays are compared
  exactly, and so are `datetime64` arrays (an instant is not a quantity), even if a tolerance is
  given. A `timedelta64` array, like a `datetime.timedelta`, follows the tolerance (`atol` is a
  number of seconds), and so does a `pandas` series or column of timedeltas
- `decimal.Decimal`, `fractions.Fraction` and `datetime.timedelta`: compared with `==`, or with their
  difference if `atol` or `rtol` is given (`atol` of a `timedelta` is a number of seconds). A
  `Decimal` NaN is equal to a NaN only if `equal_nan=True`
- `set` and `frozenset`: the elements are matched by hash and type in linear time, so `equal_nan`
  applies to the elements (`{1.0, float("nan")}` equals itself with `equal_nan=True`, also inside
  tuples and frozensets), but the tolerance does not: the elements are compared exactly, so
  `{1.0}` and `{1.0 + 1e-9}` are different even with `objects_are_allclose`
- Any other type, for example `str`, `bytes`, `range` or `datetime.datetime`:
  compared with `==`, so the tolerance and `equal_nan` are ignored, and the result is what the
  type's `__eq__` returns. If `==` raises a `TypeError` or a `ValueError`, or does not return a
  value that can be converted to a `bool` (for example, a class that compares arrays), a
  `TypeError` is raised: register a tester for this type
- Objects with a reference cycle (a list that contains itself) are supported: a pair of objects that
  is already being compared is considered equal, so two cyclic structures are equal if they unfold to
  the same values. Very deep (but not cyclic) structures raise a `MaxDepthExceededError` beyond
  `max_depth`, never a `RecursionError`
- Two objects are only compared if they have exactly the same type: a subclass is not equal to
  its parent class, and `numpy.float32(1)` is not equal to `numpy.float64(1)`

### Exact Equality Rules

#### `object`

By default, two objects are equal if:

- they have the same type
- they are equal i.e. `actual == expected` returns `True`

**Example**

```pycon
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(1, 1)
True
>>> objects_are_equal(1, 2)
False
>>> objects_are_equal(1, 1.0)
False
>>> objects_are_equal(True, True)
True
>>> objects_are_equal("abc", "abcd")
False

```

#### `collections.abc.Mapping` | `dict` | `collections.OrderedDict`

Two `Mapping`s are equal if:

- they have the same type
- they have the same number of elements i.e. `len(mapping1) == len(mapping2)` returns `True`
- they have the same set of keys i.e. `set(mapping1.keys()) == set(mapping2.keys())` returns `True`
- For each key, the values are equal. The value associated to the key `k` in the first mapping has
  to be equal to value associated to the key `k` in the second mapping.
- For an `OrderedDict` only: the keys are in the same order. The order is ignored for the other
  mappings.

```pycon
>>> from collections import OrderedDict
>>> from coola.equality import objects_are_equal
>>> objects_are_equal({"int": 1, "str": "abc"}, {"int": 1, "str": "abc"})
True
>>> objects_are_equal({"int": 1, "str": "abc"}, OrderedDict({"int": 1, "str": "abc"}))
False
>>> objects_are_equal({"int": 1, "str": "abc"}, {"int": 1, "str": "abc", "float": 0.2})
False
>>> objects_are_equal({"int": 1, "str": "abc"}, {"int": 1, "float": 0.2})
False
>>> objects_are_equal({"int": 1, "str": "abc"}, {"int": 1, "str": "abcd"})
False

```

#### `collections.abc.Sequence` | `list` | `tuple`

Two `Sequence`s are equal if:

- they have the same type
- they have the same number of elements i.e. `len(sequence1) == len(sequence2)` returns `True`
- For each position, the elements are equal. The `i`-th element in the first sequence has to be
  equal to the `i`-th element in the second sequence.

**Example**

```pycon
>>> from coola.equality import objects_are_equal
>>> objects_are_equal([1, 2, "abc"], [1, 2, "abc"])
True
>>> objects_are_equal([1, 2, "abc"], (1, 2, "abc"))
False
>>> objects_are_equal([1, 2, "abc"], [1, 2, "abc", 4])
False
>>> objects_are_equal([1, 2, "abc"], [1, 2, "abcd"])
False

```

#### PyTorch

You need to install `coola` with PyTorch to check if some PyTorch objects are equal or not. Please
check the [get started page](../get_started.md) for more information. `coola` currently support the
following PyTorch objects:

- `torch.Tensor`
- `torch.nn.utils.rnn.PackedSequence`

##### `torch.Tensor`

Two PyTorch `Tensor`s are equal if:

- they have the same data type i.e. `tensor1.dtype == tensor2.dtype` returns `True`
- they have the same device i.e. `tensor1.device == tensor2.device` returns `True`
- they have the same shape i.e. `tensor1.shape == tensor2.shape` returns `True`
- they have the same values i.e. `tensor1.equal(tensor2)` returns `True`

**Example**

```pycon
>>> import torch
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(torch.ones(2, 3), torch.ones(2, 3))
True
>>> objects_are_equal(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))
False
>>> objects_are_equal(torch.ones(2, 3), torch.zeros(2, 3))
False
>>> objects_are_equal(torch.ones(2, 3), torch.ones(6))
False

```

##### `torch.nn.utils.rnn.PackedSequence`

Two PyTorch `PackedSequence`s are equal if:

- The `data` attributes are equal
- The `batch_sizes` attributes are equal
- The `sorted_indices` attributes are equal
- The `unsorted_indices` attributes are equal

**Example**

```pycon
>>> import torch
>>> from torch.nn.utils.rnn import pack_padded_sequence
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
... )
True
>>> objects_are_equal(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).add(1).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
... )  # different values
False
>>> objects_are_equal(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 2], dtype=torch.long),
...         batch_first=True,
...     ),
... )  # different lengths
False

```

#### NumPy Arrays

You need to install `coola` with NumPy to check if some NumPy ndarrays are equal or not. Please
check the [get started page](../get_started.md) for more information.

##### `numpy.ndarray`

Two NumPy `ndarray`s are equal if:

- they have the same data type i.e. `array1.dtype == array2.dtype` returns `True`
- they have the same shape i.e. `array1.shape == array2.shape` returns `True`
- they have the same values i.e. `numpy.array_equal(array1, array2)` returns `True`

```pycon
>>> import numpy
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(numpy.ones((2, 3)), numpy.ones((2, 3)))
True
>>> objects_are_equal(numpy.ones((2, 3)), numpy.ones((2, 3), dtype=int))
False
>>> objects_are_equal(numpy.ones((2, 3)), numpy.zeros((2, 3)))
False
>>> objects_are_equal(numpy.ones((2, 3)), numpy.ones((6,)))
False

```

#### pandas DataFrames and Series

pandas DataFrames and Series are equal if they have the same shape, dtypes, index and columns
(class, names, dtype and values; the tolerance never applies to an index), and the same values.
The columns are compared one by one, and the comparison stops at the first column that differs.
The numeric and duration columns follow the numeric rule (tolerance and `equal_nan`); the other
columns, including the nested and `object` columns, are compared by the `equals` of pandas.

The time resolution is part of the dtype, as in NumPy: `timedelta64[s]` and `timedelta64[ns]` (or
`datetime64[s]` and `datetime64[ns]`) are different dtypes, so two frames that hold the same instants
with different resolutions are different. pandas 3 infers the resolution from how the values are
built, so cast both sides with `astype` to the resolution you want before the comparison.

```pycon
>>> import pandas as pd
>>> from coola.equality import objects_are_equal
>>> df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
>>> df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
>>> objects_are_equal(df1, df2)
True

```

#### xarray

You need to install `coola` with xarray to check if some xarray objects are equal or not. Please
check the [get started page](../get_started.md) for more information. `coola` currently support the
following xarray objects:

- `xarray.DataArray`
- `xarray.Dataset`

##### `xarray.DataArray`

Two xarray `DataArray`s are equal if:

- they have the same data values (attribute `data`)
- they have the same name (attribute `name`)
- they have the same dimension names (attribute `dims`)
- they have the same coordinates (attribute `coords`)
- they have the same metadata (attribute `attrs`)

Unlike `xarray.DataArray.identical`, two `DataArray`s are not equal if both objects have NaNs in the
same positions to follow the standard usage in numpy.
You can use `objects_are_allclose` to compare objects with NaNs.

```pycon
>>> import numpy as np
>>> import xarray as xr
>>> from coola.equality import objects_are_equal
>>> objects_are_equal(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"]))
True
>>> objects_are_equal(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.zeros(6), dims=["z"]))
False

```

##### `xarray.Dataset`

Two xarray `Dataset`s are equal if `xarray.Dataset.identical` returns `True`.
In contrast to the standard usage in numpy, NaNs are compared like numbers, two `Dataset`s are equal
if both objects have NaNs in the same positions.

```pycon
>>> import numpy as np
>>> import xarray as xr
>>> from coola.equality import objects_are_equal
>>> ds1 = xr.Dataset(
...     {
...         "x": xr.DataArray(
...             np.arange(6),
...             dims=["z"],
...         ),
...         "y": xr.DataArray(
...             np.ones((6, 3)),
...             dims=["z", "t"],
...         ),
...     },
...     coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
... )
>>> ds2 = xr.Dataset(
...     {
...         "x": xr.DataArray(
...             np.arange(6),
...             dims=["z"],
...         ),
...         "y": xr.DataArray(
...             np.ones((6, 3)),
...             dims=["z", "t"],
...         ),
...     },
...     coords={"z": np.arange(6) + 1, "t": ["t1", "t2", "t3"]},
... )
>>> ds3 = xr.Dataset(
...     {
...         "x": xr.DataArray(
...             np.arange(6),
...             dims=["z"],
...         ),
...     },
...     coords={"z": np.arange(6) + 1},
... )
>>> objects_are_equal(ds1, ds2)
True
>>> objects_are_equal(ds1, ds3)
False

```

### Tolerance-Based Equality Rules

#### `object`

The concept of equal within a tolerance does not make sense for all `object`s.
In general, the tolerance is not used for `object`s.
The tolerance is only used for numbers (see below).
By default, two objects are equal if:

- they have the same type
- they are equal i.e. `actual == expected` returns `True`

**Example**

```pycon
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose("abc", "abc")
True
>>> objects_are_allclose("abc", "abcd")
False

```

#### Numbers: `bool` | `int` | `float` | `complex` | NumPy scalars

Two numbers are equal within a tolerance if:

- they have the same type
- the values are equal with a tolerance

**Example**

```pycon
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(1, 2)
False
>>> objects_are_allclose(1, 2, atol=1)
True
>>> objects_are_allclose(1, 2, rtol=1)
True
>>> objects_are_allclose(1.0, 2.0)
False
>>> objects_are_allclose(1.0, 2.0, atol=1)
True
>>> objects_are_allclose(1.0, 2.0, rtol=1)
True
>>> objects_are_allclose(1, 2.0, atol=1)
False

```

The tolerance also applies to `complex` numbers (the distance in the complex plane is used) and to
the NumPy scalars like `numpy.float32` or `numpy.int64`. Two numbers of different types are never
equal, for example `numpy.float32(1.0)` and `numpy.float64(1.0)`.

The tolerance is computed in double precision for the arrays: `float16`, `bfloat16` and `float32`
are cast to `float64`, so the tolerance cannot overflow at the edge of their range. The difference of
two integers is exact, whatever the size of the integers (`2**62` and `2**62 + 1` are not within
`atol=0.5`); only its comparison to the tolerance is made in `float64`. In a rare case, a difference
above 2^53 that is exactly equal to the tolerance can be rounded the other way. JAX without 64-bit
mode has no `float64` and uses the widest dtype it has.

Note that a boolean is a label, not a quantity: the tolerance never applies to booleans (Python
`bool`, `numpy.bool_`, boolean arrays and tensors, pandas boolean Series, DataFrame columns, Index
and arrays) or to enum members (`IntEnum` and `IntFlag` included), even if `bool` is a subclass of
`int`:

```pycon
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(True, False)
False
>>> objects_are_allclose(True, False, atol=1)
False
>>> objects_are_allclose(True, False, rtol=1)
False
>>> objects_are_allclose(True, True, atol=1)
True

```

#### `collections.abc.Mapping` | `dict` | `collections.OrderedDict`

Two `Mapping`s are equal within a tolerance if:

- they have the same type
- they have the same number of elements i.e. `len(mapping1) == len(mapping2)` returns `True`
- they have the same set of keys i.e. `set(mapping1.keys()) == set(mapping2.keys())` returns `True`
- For each key, the values are equal within a tolerance. The value associated to the key `k` in the
  first mapping has to be equal within the tolerance to value associated to the key `k` in the
  second mapping.
- For an `OrderedDict` only: the keys are in the same order. The tolerance does not apply to the
  keys.

**Example**

```pycon
>>> from collections import OrderedDict
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 1, "str": "abc"})
True
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 2, "str": "abc"}, atol=2)
True
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 2, "str": "abc"}, rtol=1)
True
>>> objects_are_allclose({"int": 1, "str": "abc"}, OrderedDict({"int": 1, "str": "abc"}))
False
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 1, "str": "abc", "float": 0.2})
False
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 1, "float": 0.2})
False
>>> objects_are_allclose({"int": 1, "str": "abc"}, {"int": 1, "str": "abcd"})
False

```

#### `collections.abc.Sequence` | `list` | `tuple`

Two `Sequence`s are equal within a tolerance if:

- they have the same type
- they have the same number of elements i.e. `len(sequence1) == len(sequence2)` returns `True`
- For each position, the elements are equal within a tolerance. The `i`-th element in the first
  sequence has to be equal within a tolerance to the `i`-th element in the second sequence.

**Example**

```pycon
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose([1, 2, "abc"], [1, 2, "abc"])
True
>>> objects_are_allclose([1, 2, "abc"], [1, 3, "abc"], atol=2)
True
>>> objects_are_allclose([1, 2, "abc"], [1, 3, "abc"], rtol=1)
True
>>> objects_are_allclose([1, 2, "abc"], (1, 2, "abc"))
False
>>> objects_are_allclose([1, 2, "abc"], [1, 2, "abc", 4])
False
>>> objects_are_allclose([1, 2, "abc"], [1, 2, "abcd"])
False

```

#### PyTorch

You need to install `coola` with PyTorch to check if some PyTorch objects are equal within a
tolerance or not. Please check the [get started page](../get_started.md) for more information. `coola`
currently support the following PyTorch objects:

- `torch.Tensor`
- `torch.nn.utils.rnn.PackedSequence`

##### `torch.Tensor`

Two PyTorch `Tensor`s are equal if:

- they have the same data type i.e. `tensor1.dtype == tensor2.dtype` returns `True`
- they have the same device i.e. `tensor1.device == tensor2.device` returns `True`
- they have the same shape i.e. `tensor1.shape == tensor2.shape` returns `True`
- the values are equal within a tolerance i.e. `tensor1.allclose(tensor2)` returns `True`

**Example**

```pycon
>>> import torch
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3))
True
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3) + 1, atol=2)
True
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3) + 1, rtol=1)
True
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(2, 3, dtype=torch.long))
False
>>> objects_are_allclose(torch.ones(2, 3), torch.zeros(2, 3))
False
>>> objects_are_allclose(torch.ones(2, 3), torch.ones(6))
False

```

##### `torch.nn.utils.rnn.PackedSequence`

Two PyTorch `PackedSequence`s are equal within a tolerance if:

- The `data` attributes are equal within a tolerance
- The `batch_sizes` attributes are equal
- The `sorted_indices` attributes are equal
- The `unsorted_indices` attributes are equal

**Example**

```pycon
>>> import torch
>>> from torch.nn.utils.rnn import pack_padded_sequence
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
... )
True
>>> objects_are_allclose(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float() + 1,
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     atol=2,
... )
True
>>> objects_are_allclose(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).add(1).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
... )  # different values
False
>>> objects_are_allclose(
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 3], dtype=torch.long),
...         batch_first=True,
...     ),
...     pack_padded_sequence(
...         input=torch.arange(10).view(2, 5).float(),
...         lengths=torch.tensor([5, 2], dtype=torch.long),
...         batch_first=True,
...     ),
... )  # different lengths
False

```

#### NumPy Arrays

You need to install `coola` with NumPy to check if some NumPy ndarrays are equal or not. Please
check the [get started page](../get_started.md) for more information.

##### `numpy.ndarray`

Two NumPy `ndarray`s are equal within a tolerance if:

- they have the same data type i.e. `array1.dtype == array2.dtype` returns `True`
- they have the same shape i.e. `array1.shape == array2.shape` returns `True`
- the values are equal within a tolerance i.e. `numpy.allclose(array1, array2)` returns `True`

```pycon
>>> import numpy
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)))
True
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)) + 1, atol=2)
True
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3)) + 1, rtol=1)
True
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((2, 3), dtype=int))
False
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.zeros((2, 3)))
False
>>> objects_are_allclose(numpy.ones((2, 3)), numpy.ones((6,)))
False

```

##### `numpy.ma.MaskedArray`

Two NumPy `MaskedArray`s are equal if:

- they have the same data type and the same shape
- the underlying data are equal (within the tolerance, if one is given), **including the masked
  values**: two arrays that differ only at a masked position are not equal
- they have the same mask
- they have the same `fill_value`

The tolerance is only used for the numerical data. The mask and the `fill_value` are compared
exactly.

```pycon
>>> import numpy
>>> from coola.equality import objects_are_equal, objects_are_allclose
>>> objects_are_equal(
...     numpy.ma.array(data=[0.0, 1.0, 2.0], mask=[0, 1, 0]),
...     numpy.ma.array(data=[0.0, 1.0, 2.0], mask=[0, 1, 0]),
... )
True
>>> objects_are_equal(
...     numpy.ma.array(data=[0.0, 1.0, 2.0], mask=[0, 1, 0]),
...     numpy.ma.array(data=[0.0, 5.0, 2.0], mask=[0, 1, 0]),
... )  # the masked values are different
False
>>> objects_are_allclose(
...     numpy.ma.array(data=[0.0, 1.0, 2.0], mask=[0, 1, 0]),
...     numpy.ma.array(data=[0.0, 1.0, 2.001], mask=[0, 1, 0]),
...     atol=0.01,
... )
True
>>> objects_are_equal(
...     numpy.ma.array(data=[0.0, 1.0, 2.0], mask=[0, 1, 0]),
...     numpy.ma.array(data=[0.0, 1.0, 2.0], mask=[1, 1, 0]),
... )  # the masks are different
False

```

#### xarray

You need to install `coola` with xarray to check if some xarray objects are equal or not. Please
check the [get started page](../get_started.md) for more information. `coola` currently support the
following xarray objects:

- `xarray.DataArray`
- `xarray.Dataset`

##### `xarray.DataArray`

Two xarray `DataArray`s are equal within a tolerance if:

- they have the same data values within the tolerance (attribute `data`)
- they have the same name (attribute `name`)
- they have the same dimension names (attribute `dims`)
- they have the same coordinates (attribute `coords`)
- they have the same metadata (attribute `attrs`)

Unlike `xarray.DataArray.identical`, two `DataArray`s are not equal if both objects have NaNs in the
same positions to follow the standard usage in numpy.
You can use `objects_are_allclose` to compare objects with NaNs.

```pycon
>>> import numpy as np
>>> import xarray as xr
>>> from coola.equality import objects_are_allclose
>>> objects_are_allclose(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.arange(6), dims=["z"]))
True
>>> objects_are_allclose(xr.DataArray(np.arange(6), dims=["z"]), xr.DataArray(np.zeros(6), dims=["z"]))
False

```

## Common Use Cases

### Testing

The equality functions are particularly useful in unit tests:

```pycon
>>> from coola.equality import objects_are_equal
>>> def process_data(data):
...     # Some data processing
...     return {"result": [x * 2 for x in data]}
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

## Behavior by Type: NaN and Tolerance

The arguments `equal_nan`, `atol` and `rtol` are not supported by every library. This table
summarizes what each type does with them (it was checked for each type with `objects_are_equal` and
`objects_are_allclose`).

| Type                                                    | `equal_nan`                                                                                 | `atol` / `rtol`                                                                                        | Also compared                                            |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| `float`, `int`, `complex`, NumPy scalars                | Honored for distinct `NaN` objects\*                                                        | `abs(a - b) <= atol + rtol * max(abs(a), abs(b))`, for `int` too (exact for large `int`)               | The types must be the same (`1` and `1.0` are different) |
| dataclass instances                                     | Honored for the fields                                                                      | Honored for the fields                                                                                 | Same type; the fields with `compare=True`                |
| `numpy.ndarray`                                         | Honored for numeric and datetime dtypes; **ignored for `str`, `bytes` and `object` arrays** | Numeric dtypes only (`bool`, `int`, `uint`, `float`, `complex`); the other dtypes are compared exactly | dtype, shape                                             |
| `numpy.ma.MaskedArray`                                  | Honored for the data (masked values included)                                               | Honored for the data (masked values included)                                                          | dtype, shape, mask, `fill_value`                         |
| `torch.Tensor`                                          | Honored                                                                                     | Honored (`torch.allclose`), integer tensors included                                                   | dtype, shape, device                                     |
| `torch.nn.utils.rnn.PackedSequence`                     | Honored                                                                                     | Honored                                                                                                | `batch_sizes`, `sorted_indices`, `unsorted_indices`      |
| `jax.numpy.ndarray`                                     | Honored                                                                                     | Honored                                                                                                | dtype, shape                                             |
| `pandas.DataFrame`, `pandas.Series`                     | A float `NaN` follows `equal_nan`; a null (`None`, `NaT`, `pd.NA`) is always equal          | Honored for the integer and floating-point columns (NumPy rule)                                        | dtype, index, columns                                    |
| `polars.DataFrame`, `polars.Series`, `polars.LazyFrame` | Honored for float `NaN`. **`null` is always equal to `null`**                               | Honored for the integer and floating-point columns (NumPy rule)                                        | dtype, schema; a `LazyFrame` is collected                |
| `xarray.DataArray`, `xarray.Dataset`, `xarray.Variable` | Honored                                                                                     | Honored                                                                                                | dimensions, coordinates, attributes                      |
| `pyarrow.Array`, `pyarrow.Table`                        | Honored for the floating-point columns; `null` is equal to `null`                           | Honored for the integer and floating-point columns; the other types are compared exactly               | Type, length, `null` positions, schema for tables        |

\* Python short-circuits on identity, so `x is x` is always equal. Two distinct `NaN` floats are
not equal unless `equal_nan=True`.

Details:

- **pandas**: with `equal_nan=False` (the default), a data frame or a series that contains a float
  (or complex) `NaN` is never equal to another one, **even to a copy of itself**, because the `NaN`
  values are not equal to each other. Use `equal_nan=True` to compare the data frames with `NaN`.
  The other missing values (`None`, `pandas.NA`, `NaT`) are equal to each other, whatever
  `equal_nan` is.
- **polars**: the shape, the column names, order and dtypes are compared first, then the columns
  one by one, stopping at the first column that differs. A float `NaN` follows `equal_nan`, and a
  `null` is equal to a `null`. The nested columns are left to polars.
- **PyArrow**: the type and the length of an array, or the schema (without its metadata) and the
  number of rows of a table, are compared first, then the columns one by one, stopping at the first
  column that differs. The integer, floating-point and duration columns are compared like NumPy
  arrays, and the `null` values must be at the same positions. The other types (strings, nested
  types, etc.) are compared exactly with `equals`. Without NumPy, `equal_nan`, `atol` and `rtol` are
  ignored and a `RuntimeWarning` is emitted. `pyarrow.Array`, `pyarrow.ChunkedArray`,
  `pyarrow.RecordBatch` and `pyarrow.Table` are supported.
- **NumPy**: `equal_nan=True` can safely be used with an object that contains arrays of strings:
  the argument is only applied to the dtypes that have a `NaN`.
- **`-0.0`** is equal to `0.0` for all the types, whatever the tolerance.

## Explaining a Difference

`compare` returns a `ComparisonResult` with the path to the difference and the reason:

```pycon
>>> from coola.equality import compare
>>> result = compare({"a": [1, 2]}, {"a": [1, 3]})
>>> result.equal
False
>>> result.path
('a', 1)
>>> print(result.format_message())
objects are not equal at actual['a'][1]
numbers are different:
  actual   : 2
  expected : 3

```

`assert_objects_equal` and `assert_objects_allclose` raise an `AssertionError` with the same
message, for the tests.

The path is read as follows: a key or a position is itself (`['a']`, `[3]`), a field or an attribute
is `.x`, an element of a set is `{2}`, and an array reports its index (`[1][2]`). A pandas, polars or
PyArrow table is reported as a whole, with the first differing column in the reason (for example
`tables have different values: column 'a' differs`); use the library (`DataFrame.compare`,
`polars.testing`, `Table.equals`) to find the row. The design behind this is described in
[Architecture](../dev/architecture.md#design-of-coolaequality).

- `compare` and the assert functions do not log and do not change the logging configuration, so they
  can be called from several threads at the same time.
- A custom tester is explained by `compare` if it returns the child pairs (the engine adds the
  path) or calls `ctx.fail(reason, *path)`. The reason can be a callable, which is called only when
  the difference is explained.
- Only the first difference is explained, and only on demand: `objects_are_equal` and
  `objects_are_allclose` build no message and no path. The values in a reason are bounded, so
  explaining a difference in a large object does not format the object.

## Design Principles

The `coola.equality` package is designed around several key principles:

1. **Type-aware**: Different types are compared using appropriate logic (e.g., NumPy arrays use
   array comparison, not object comparison)

2. **Recursive**: Nested structures are compared recursively, so you can compare complex data
   structures

3. **Extensible**: The registry system allows adding support for new types without modifying the
   core library

4. **Configurable**: Parameters like `atol`, `rtol` and `equal_nan` give you
   fine-grained control

5. **Consistent**: The same comparison logic is used whether objects are at the top level or nested
   deep in a structure

6. **Fast**: Performance is a requirement, not a nice-to-have: a comparison that is too slow is not
   useful, so a new feature or a new type must not make the common cases slower, and the speed of
   the passing and of the failing comparison are both measured by the benchmarks

7. **Stops at the first difference**: As soon as two objects are found to be different, the
   comparison stops and returns `False`; the rest of the objects is not visited

8. **Explains the first difference only, on demand**: The explanation (`compare`, `assert_*`) is the
   path and the reason of the **first** difference found. Nothing is formatted by
   `objects_are_equal` and `objects_are_allclose`, which only return a `bool`

## See Also

- [`coola.equality.tester`](equality_tester.md): Learn about the tester registry system

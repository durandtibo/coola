# Equality Testers

:book: This page describes the `coola.equality.tester` package, which provides the registry-based
system for dispatching equality comparisons to type-specific testers.

**Prerequisites:** You'll need to know a bit of Python and understand the basics of the
[coola.equality](equality.md) package.
For a refresher on Python, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.equality.tester` package provides a registry system that manages equality testers for
different types. When you call `objects_are_equal()` or `objects_are_allclose()`, the registry
automatically selects the appropriate tester based on the type of the objects being compared.

Key components:

- **`BaseEqualityTester`**: Abstract base class for all equality testers
- **`EqualityTesterRegistry`**: Registry that dispatches to appropriate testers by type
- **Type-specific testers**: Specialized testers for Python built-ins and third-party libraries
- **Default registry**: Pre-configured registry with testers for common types

A tester never recurses: it returns a verdict or the child pairs that must all be equal, and the
`Engine` compares them with an explicit stack.

## Key Concepts

### Equality Testers

An equality tester is responsible for comparing objects of a specific type. Each tester:

1. Checks if it can handle the given type
2. Returns `True` if the objects are equal, `False` if they are different (`ctx.fail(reason)`), or
   the child pairs that must all be equal

```pycon
>>> from coola.equality import objects_are_equal
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import SequenceEqualityTester, get_default_registry
>>> engine = Engine(EqualityConfig())
>>> tester = SequenceEqualityTester()
>>> list(tester.objects_are_equal([1.0, 2.0], [1.0, 3.0], engine))
[(1.0, 1.0, 0), (2.0, 3.0, 1)]
>>> engine.run([1, 2, 3], [1, 2, 3])
True
>>> engine.run([1, 2, 3], [1, 2, 4])
False

```

### Registry System

!!! tip "Import collisions"

    `get_default_registry` exists in several `coola` subpackages with different return types. Import it
    from `coola.equality.tester`, or import the module and call `tester.get_default_registry()`.

The registry maintains a mapping from types to testers and automatically selects the right tester:

```pycon
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import get_default_registry
>>> registry = get_default_registry()
>>> config = EqualityConfig()
>>> # Registry automatically selects SequenceEqualityTester for lists
>>> registry.objects_are_equal([1, 2, 3], [1, 2, 3], config)
True
>>> # Registry automatically selects MappingEqualityTester for dicts
>>> registry.objects_are_equal({"a": 1}, {"a": 1}, config)
True

```

The registry uses Python's Method Resolution Order (MRO) to find the most specific tester for a
given type, which means it can handle inheritance correctly.

## Built-in Testers

### Collection Testers

#### `SequenceEqualityTester`

Handles sequence types like `list`, `tuple`, and `deque`:

```pycon
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> engine.run([1, 2, 3], [1, 2, 3])
True
>>> engine.run([1, 2, 3], [1, 2, 4])
False
>>> # Works with nested structures
>>> engine.run(
...     [1, {"a": 2}, [3, 4]],
...     [1, {"a": 2}, [3, 4]],
... )
True

```

The tester recursively compares elements using the registry, so nested structures are handled
correctly.

#### `SetEqualityTester`

Handles `set` and `frozenset`. Unlike `==`, the elements are matched by hash and type, so
`equal_nan` applies to them (the tolerance does not):

```pycon
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry
>>> config = EqualityConfig(equal_nan=True)
>>> engine = Engine(config)
>>> engine.run({1.0, float("nan")}, {1.0, float("nan")})
True
>>> engine.run({1.0, 2.0}, {1.0, 3.0})
False

```

#### `MappingEqualityTester`

Handles mapping types like `dict`:

```pycon
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> engine.run({"a": 1, "b": 2}, {"a": 1, "b": 2})
True
>>> engine.run({"a": 1, "b": 2}, {"a": 1, "b": 3})
False
>>> # Works with nested structures
>>> engine.run(
...     {"x": [1, 2], "y": {"z": 3}},
...     {"x": [1, 2], "y": {"z": 3}},
... )
True

```

#### `OrderedMappingEqualityTester`

Like `MappingEqualityTester`, and it also requires the keys to be in the same order. It is
registered for `collections.OrderedDict`, whose order is meaningful (Python's `==` also compares it
between two `OrderedDict`s). The other mappings ignore the order of the keys:

```pycon
>>> from collections import OrderedDict
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> engine.run(OrderedDict(a=1, b=2), OrderedDict(a=1, b=2))
True
>>> engine.run(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1))
False

```

### Scalar Testers

#### `ScalarEqualityTester`

Handles scalar types like `int`, `float`, and `bool`:

```pycon
>>> from coola.equality import Tolerance
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry, ScalarEqualityTester
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> tester = ScalarEqualityTester()
>>> tester.objects_are_equal(42, 42, engine)
True
>>> tester.objects_are_equal(3.14, 3.14, engine)
True
>>> # With tolerance
>>> engine_tol = Engine(EqualityConfig(tolerance=Tolerance(atol=0.01)))
>>> tester.objects_are_equal(1.0, 1.005, engine_tol)
True

```

#### `EqualEqualityTester`

Generic tester for objects that have an `equal(other)` method. The objects are equal if
`actual.equal(expected)` is true; an object without an `equal` method is not equal. It is not
registered by default: register it for your own types:

```pycon
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import EqualEqualityTester
>>> class MyList(list):
...     def equal(self, other: object) -> bool:
...         return self == other
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> tester = EqualEqualityTester()
>>> tester.objects_are_equal(MyList([1, 2, 3]), MyList([1, 2, 3]), engine)
True
>>> tester.objects_are_equal(MyList([1, 2, 3]), MyList([1, 2, 4]), engine)
False

```

#### `DifferenceCloseEqualityTester`

`decimal.Decimal`, `fractions.Fraction` and `datetime.timedelta` have no `allclose` method, so this
tester compares their difference with the tolerance. A `Decimal` NaN is equal to a NaN only if
`equal_nan` is `True`, and the `atol` of a `timedelta` is a number of seconds. It is registered for
these three types in the default registry:

```pycon
>>> from decimal import Decimal
>>> from coola.equality import Tolerance
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import DifferenceCloseEqualityTester
>>> tester = DifferenceCloseEqualityTester()
>>> tester.objects_are_equal(Decimal("1.0"), Decimal("1.00001"), Engine(EqualityConfig()))
False
>>> config = EqualityConfig(tolerance=Tolerance(atol=1e-3))
>>> tester.objects_are_equal(Decimal("1.0"), Decimal("1.00001"), Engine(config))
True

```

### NumPy Testers

#### `NumpyArrayEqualityTester`

Handles NumPy arrays:

```pycon
>>> from coola.equality import Tolerance
>>> import numpy as np
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry, NumpyArrayEqualityTester
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> tester = NumpyArrayEqualityTester()
>>> tester.objects_are_equal(
...     np.array([1, 2, 3]),
...     np.array([1, 2, 3]),
...     engine,
... )
True
>>> tester.objects_are_equal(
...     np.array([1, 2, 3]),
...     np.array([1, 2, 4]),
...     engine,
... )
False
>>> # With tolerance
>>> engine_tol = Engine(EqualityConfig(tolerance=Tolerance(atol=0.01)))
>>> tester.objects_are_equal(
...     np.array([1.0, 2.0, 3.0]),
...     np.array([1.0, 2.0, 3.001]),
...     engine_tol,
... )
True

```

#### `NumpyMaskedArrayEqualityTester`

Handles NumPy masked arrays:

```pycon
>>> import numpy as np
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry, NumpyMaskedArrayEqualityTester
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> tester = NumpyMaskedArrayEqualityTester()
>>> arr1 = np.ma.array([1, 2, 3], mask=[0, 0, 1])
>>> arr2 = np.ma.array([1, 2, 99], mask=[0, 0, 1])  # Masked value differs
>>> engine.run(arr1, arr2)
False

```

#### `NumpyTimeScalarEqualityTester`

Handles `numpy.timedelta64` and `numpy.datetime64` scalars. `NaT` is equal to `NaT` only if
`equal_nan` is `True`. A `timedelta64` is a duration, so the tolerances apply (`atol` is a number
of seconds), whereas a `datetime64` is an instant and is compared exactly:

```pycon
>>> import numpy as np
>>> from coola.equality import Tolerance
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import NumpyTimeScalarEqualityTester
>>> tester = NumpyTimeScalarEqualityTester()
>>> engine = Engine(EqualityConfig(tolerance=Tolerance(atol=1.0)))
>>> tester.objects_are_equal(np.timedelta64(1, "s"), np.timedelta64(2000, "ms"), engine)
True
>>> tester.objects_are_equal(np.datetime64("2020-01-01"), np.datetime64("2020-01-02"), engine)
False

```

### PyTorch Testers

#### `TorchTensorEqualityTester`

Handles PyTorch tensors:

```pycon
>>> from coola.equality import Tolerance
>>> import torch
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry, TorchTensorEqualityTester
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> tester = TorchTensorEqualityTester()
>>> tester.objects_are_equal(
...     torch.tensor([1, 2, 3]),
...     torch.tensor([1, 2, 3]),
...     engine,
... )
True
>>> # With tolerance
>>> engine_tol = Engine(EqualityConfig(tolerance=Tolerance(atol=0.01)))
>>> tester.objects_are_equal(
...     torch.tensor([1.0, 2.0, 3.0]),
...     torch.tensor([1.0, 2.0, 3.001]),
...     engine_tol,
... )
True

```

#### `TorchPackedSequenceEqualityTester`

Handles PyTorch PackedSequence objects:

```pycon
>>> import torch
>>> from torch.nn.utils.rnn import pack_padded_sequence
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.engine import Engine
>>> from coola.equality.tester import get_default_registry, TorchPackedSequenceEqualityTester
>>> config = EqualityConfig()
>>> engine = Engine(config)
>>> tester = TorchPackedSequenceEqualityTester()
>>> ps1 = pack_padded_sequence(
...     torch.arange(10).view(2, 5).float(),
...     torch.tensor([5, 3], dtype=torch.long),
...     batch_first=True,
... )
>>> ps2 = pack_padded_sequence(
...     torch.arange(10).view(2, 5).float(),
...     torch.tensor([5, 3], dtype=torch.long),
...     batch_first=True,
... )
>>> engine.run(ps1, ps2)
True

```

### Table Testers

pandas, polars and PyArrow run the same algorithm, written once in `TableEqualityTester`: the
structure is compared first (shape, names, data types, index), then the columns one by one, each by
its kind, and the comparison stops at the first column that differs. An adapter converts an object
of a library to a table. The default registry registers one `TableEqualityTester` per type:

| Library | Types                                                                |
| ------- | -------------------------------------------------------------------- |
| pandas  | `DataFrame`, `Series`, `Index` (also `MultiIndex`), extension arrays |
| polars  | `DataFrame`, `LazyFrame`, `Series`                                   |
| PyArrow | `Array`, `ChunkedArray`, `RecordBatch`, `Table`                      |

The numbers (integers, floats, durations) go through the numeric kernel: the tolerance and
`equal_nan` apply and the null values must be at the same positions. The other columns (booleans,
strings, categories, instants, decimals, nested values and `object` columns) are compared by the
library (`equals`), which never sees the tolerance or `equal_nan`.

Without NumPy, the PyArrow columns are compared by Arrow alone: the tolerance and `equal_nan` cannot
be applied, so a `RuntimeWarning` says that they are ignored (once per comparison, through
`ctx.warn_once`).

A table is one leaf for the engine: `compare` reports it as different, with the name of the first
column that differs and no row. Use the tools of the library to find the row.

```pycon
>>> import pandas as pd
>>> from coola.equality import compare
>>> df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
>>> df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
>>> result = compare({"prices": df1}, {"prices": df2})
>>> result.path
('prices',)
>>> print(result.reason)
tables have different values: column 'b' differs

```

To support another library, write an adapter: a function that returns an object with a `kind_name`
(the plural word used in the reasons), a `same_structure(other, ctx)` method and a `columns()`
method that yields `Column`s, and register `TableEqualityTester(adapter)` for the type.

### Testers for Objects with Methods

- **`DataclassEqualityTester`**: Compare two instances of the same dataclass. The fields are
  compared through the registry, so they support the tolerance and `equal_nan`
- **`EqualNanEqualityTester`**: Compare objects that have an `equal(other, equal_nan)` method
- **`TolerantEqualEqualityTester`**: Compare objects that have both an `allclose` method and an
  `equal` method. `allclose` is used when `atol` or `rtol` is not zero, otherwise `equal`

### Other Library Testers

- **`JaxArrayEqualityTester`**: Compare JAX arrays
- **`XarrayDataArrayEqualityTester`**: Compare xarray DataArrays
- **`XarrayDatasetEqualityTester`**: Compare xarray Datasets
- **`XarrayVariableEqualityTester`**: Compare xarray Variables
- **`PydanticModelEqualityTester`**: Compare pydantic models. The models must have the same type,
  the same fields and the same extra values. The private attributes are not compared, and the
  location of a difference is the name of the field (or `model_extra`)
- **`ScipySparseEqualityTester`**: Compare `scipy.sparse` matrices and arrays. The class (so the
  format), the shape, the dtype and the non-zero positions must be equal, then the stored values
  are compared with the tolerance and `equal_nan`. Duplicate entries are summed and explicitly
  stored zeros are ignored, so two objects that represent the same matrix are equal

### Tester for Objects Without Value Semantics

#### `ObjectAttributesEqualityTester`

A plain class that does not define `__eq__`, an exception such as `ValueError("a")`, and a
`functools.partial` are compared by identity: two equal-looking instances are different, unlike a
dataclass or a pydantic model, which are compared field by field. This is the rule of "equal means
`==`", so the tester is **not registered by default**. Register it for the classes that must be
compared by their attributes. It compares the entries of `vars(obj)`, the slots of the class and of its
parent classes, the `args` of an exception and the `func`, `args` and `keywords` of a partial, through
the registry (so the tolerance and `equal_nan` apply), and reports the attribute that differs:

```pycon
>>> from coola.equality import objects_are_equal
>>> from coola.equality.tester import (
...     DefaultEqualityTester,
...     EqualityTesterRegistry,
...     ObjectAttributesEqualityTester,
... )
>>> class Point:
...     def __init__(self, x):
...         self.x = x
>>> registry = EqualityTesterRegistry({object: DefaultEqualityTester()})
>>> objects_are_equal(Point(1), Point(1), registry=registry)
False
>>> registry.register(Point, ObjectAttributesEqualityTester())
>>> objects_are_equal(Point(1), Point(1), registry=registry)
True

```

To change the default registry, use `register_equality_testers`, for example
`register_equality_testers({BaseException: ObjectAttributesEqualityTester()})` to compare all the
exceptions by their `args`.

### Default Tester

#### `DefaultEqualityTester`

The fallback tester used when no type-specific tester is registered:

```pycon
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import DefaultEqualityTester
>>> config = EqualityConfig()
>>> tester = DefaultEqualityTester()
>>> # Works with any type using == comparison
>>> tester.objects_are_equal("hello", "hello", config)
True
>>> tester.objects_are_equal(42, 42, config)
True

```

## Using the Registry

### Default Registry

Get the default pre-configured registry:

```pycon
>>> from coola.equality.tester import get_default_registry
>>> from coola.equality.config import EqualityConfig
>>> registry = get_default_registry()
>>> config = EqualityConfig()
>>> # Use the registry to compare objects
>>> registry.objects_are_equal([1, 2, 3], [1, 2, 3], config)
True

```

The default registry includes testers for all supported types.

### Registering Custom Testers

You can register custom testers for your own types:

```pycon
>>> from coola.equality.tester import register_equality_testers, DefaultEqualityTester
>>> # Create a custom class
>>> class MyClass:
...     def __init__(self, value):
...         self.value = value
>>> # Register a tester for MyClass (modifies the global default registry)
>>> # Note: This is skipped in doctests to avoid side effects on the global state
>>> register_equality_testers({MyClass: DefaultEqualityTester()})  # doctest: +SKIP

```

For more complex types, you would create a custom tester class (see "Creating Custom Testers"
section below).

### Creating Custom Registries

You can create a custom registry with specific testers:

```pycon
>>> from coola.equality.tester import (
...     EqualityTesterRegistry,
...     SequenceEqualityTester,
...     MappingEqualityTester,
...     DefaultEqualityTester,
... )
>>> from coola.equality.config import EqualityConfig
>>> registry = EqualityTesterRegistry(
...     {
...         list: SequenceEqualityTester(),
...         dict: MappingEqualityTester(),
...         object: DefaultEqualityTester(),
...     }
... )
>>> config = EqualityConfig(registry=registry)
>>> registry.objects_are_equal([1, 2], [1, 2], config)
True

```

## Creating Custom Testers

To create a custom tester, inherit from `BaseEqualityTester` and implement `objects_are_equal`. It
receives two objects of the same type and the `Engine` (`ctx`). For a leaf, return a verdict:

```pycon
>>> from coola.equality import objects_are_equal
>>> from coola.equality.tester import BaseEqualityTester, register_equality_testers
>>> class Version:
...     def __init__(self, major: int, minor: int) -> None:
...         self.major = major
...         self.minor = minor
>>> class VersionEqualityTester(BaseEqualityTester[Version]):
...     def objects_are_equal(self, actual: Version, expected: Version, ctx) -> bool:
...         return (actual.major, actual.minor) == (expected.major, expected.minor) or ctx.fail(
...             lambda: (
...                 f"versions differ: {actual.major}.{actual.minor} vs {expected.major}.{expected.minor}"
...             )
...         )

```

`ctx.fail(reason, *path)` records the reason and returns `False`. A reason can be a callable: it is
called only when the difference is explained (`compare`, `assert_*`), so `objects_are_equal` formats
nothing. For a container, return the child pairs `(actual, expected, path part)`; they are produced
lazily and in the order the first difference is wanted. The path part is a key or an index
(`0`, `"a"`), an `AttributeName` for an attribute, or an `Element` for an element of a set; it is
what `compare` reports in `result.path`. The engine compares them, applies the
tolerance, detects the cycles and locates the difference:

```pycon
>>> from coola.equality import AttributeName, compare
>>> class Point:
...     def __init__(self, x: float, y: float) -> None:
...         self.x = x
...         self.y = y
>>> class PointEqualityTester(BaseEqualityTester[Point]):
...     def objects_are_equal(self, actual: Point, expected: Point, ctx):
...         return (
...             (actual.x, expected.x, AttributeName("x")),
...             (actual.y, expected.y, AttributeName("y")),
...         )
>>> register_equality_testers({Point: PointEqualityTester()})  # doctest: +SKIP

```

Two testers are equal if they have the same type and the same attributes (`==`); a tester has no
state but its configuration.

## Advanced Usage

### Type Resolution with MRO

The registry uses Python's Method Resolution Order (MRO) to find testers:

```pycon
>>> from coola.equality.tester import get_default_registry
>>> from coola.equality.config import EqualityConfig
>>> from collections import OrderedDict
>>> registry = get_default_registry()
>>> config = EqualityConfig()
>>> # OrderedDict inherits from dict, but it has its own, more specific tester
>>> # And the tester checks that both objects have the same type
>>> od = OrderedDict([("a", 1), ("b", 2)])
>>> registry.objects_are_equal(od, {"a": 1, "b": 2}, config)
False
>>> # Comparing two OrderedDicts works (same type, same order)
>>> registry.objects_are_equal(od, OrderedDict([("a", 1), ("b", 2)]), config)
True
>>> registry.objects_are_equal(od, OrderedDict([("b", 2), ("a", 1)]), config)
False

```

This means you can register a tester for a base class and it will work for all subclasses.

### Recursive Comparison

A tester does not recurse: a tester for a container returns the child pairs, and the `Engine`
compares them with the registry, so the nested values get the same tolerance and `equal_nan`:

```pycon
>>> from coola.equality.tester import get_default_registry
>>> from coola.equality.config import EqualityConfig
>>> registry = get_default_registry()
>>> config = EqualityConfig()
>>> # Nested structure: list containing dict containing list
>>> nested1 = [1, {"a": [2, 3]}, 4]
>>> nested2 = [1, {"a": [2, 3]}, 4]
>>> registry.objects_are_equal(nested1, nested2, config)
True

```

The nesting depth is limited by the `max_depth` parameter in `EqualityConfig`: a deeper structure
raises a `MaxDepthExceededError`.

### Performance Considerations

The registry caches the result of the type lookups (a bounded cache that is emptied when the
registry changes) to optimize performance:

```pycon
>>> from coola.equality.tester import get_default_registry
>>> registry = get_default_registry()
>>> # First lookup for a type will cache the result
>>> # Subsequent lookups for the same type will be faster
>>> # This is automatically managed internally

```

## Common Use Cases

### Testing Framework Integration

Testers are commonly used in testing frameworks. For the common case, `assert_objects_equal` and
`assert_objects_allclose` already raise an `AssertionError` with the path and the reason of the
first difference. To build your own assertion from a registry:

```pycon
>>> from coola.equality.tester import get_default_registry
>>> from coola.equality.config import EqualityConfig
>>> def assert_equal(actual, expected):
...     registry = get_default_registry()
...     config = EqualityConfig()
...     if not registry.objects_are_equal(actual, expected, config):
...         raise AssertionError(f"{actual} != {expected}")
>>> assert_equal([1, 2, 3], [1, 2, 3])  # Passes
>>> # assert_equal([1, 2, 3], [1, 2, 4])  # Would raise AssertionError

```

### Custom Type Support

Add support for your custom types:

```pycon
>>> from coola.equality.tester import BaseEqualityTester, register_equality_testers
>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
>>> class PointEqualityTester(BaseEqualityTester[Point]):
...     def objects_are_equal(self, actual, expected, ctx):
...         return ((actual.x, expected.x, 0), (actual.y, expected.y, 1))
>>> # Register the tester (modifies global state, so skipped in doctests)
>>> register_equality_testers({Point: PointEqualityTester()})  # doctest: +SKIP

```

### Tolerance-Based Comparisons

Use testers with tolerance configurations:

```pycon
>>> from coola.equality import Tolerance
>>> import numpy as np
>>> from coola.equality.tester import get_default_registry
>>> from coola.equality.config import EqualityConfig
>>> registry = get_default_registry()
>>> config = EqualityConfig(tolerance=Tolerance(atol=1e-5, rtol=1e-5))
>>> registry.objects_are_equal(
...     np.array([1.0, 2.0, 3.0]),
...     np.array([1.00001, 2.00001, 3.00001]),
...     config,
... )
True

```

## Design Principles

The tester system is designed around several key principles:

1. **Type-based dispatch**: Automatically selects the right tester based on object type

2. **Extensibility**: Easy to add support for new types by registering custom testers

3. **One extension point**: A tester is a class with one method, and it never recurses

4. **Iterative comparison**: Nested structures are compared with an explicit stack, so there is no
   `RecursionError`

5. **Performance**: Caching of type lookups for efficient repeated comparisons

6. **Configurability**: `EqualityConfig` provides fine-grained control over comparison behavior

## See Also

- [`coola.equality`](equality.md): Main equality comparison functions
- [`coola.registry`](registry.md): General registry pattern used by the tester system

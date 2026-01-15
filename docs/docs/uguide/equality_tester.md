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

The tester system uses handler chains (from `coola.equality.handler`) to perform the actual
comparisons.

## Key Concepts

### Equality Testers

An equality tester is responsible for comparing objects of a specific type. Each tester:

1. Checks if it can handle the given type
2. Uses a handler chain to perform the comparison
3. Returns `True` if objects are equal, `False` otherwise

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import SequenceEqualityTester
>>> config = EqualityConfig()
>>> tester = SequenceEqualityTester()
>>> tester.objects_are_equal([1, 2, 3], [1, 2, 3], config)
True
>>> tester.objects_are_equal([1, 2, 3], [1, 2, 4], config)
False

```

### Registry System

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
>>> from coola.equality.tester import SequenceEqualityTester
>>> config = EqualityConfig()
>>> tester = SequenceEqualityTester()
>>> tester.objects_are_equal([1, 2, 3], [1, 2, 3], config)
True
>>> tester.objects_are_equal([1, 2, 3], [1, 2, 4], config)
False
>>> # Works with nested structures
>>> tester.objects_are_equal(
...     [1, {"a": 2}, [3, 4]],
...     [1, {"a": 2}, [3, 4]],
...     config,
... )
True

```

The tester recursively compares elements using the registry, so nested structures are handled
correctly.

#### `MappingEqualityTester`

Handles mapping types like `dict`:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import MappingEqualityTester
>>> config = EqualityConfig()
>>> tester = MappingEqualityTester()
>>> tester.objects_are_equal({"a": 1, "b": 2}, {"a": 1, "b": 2}, config)
True
>>> tester.objects_are_equal({"a": 1, "b": 2}, {"a": 1, "b": 3}, config)
False
>>> # Works with nested structures
>>> tester.objects_are_equal(
...     {"x": [1, 2], "y": {"z": 3}},
...     {"x": [1, 2], "y": {"z": 3}},
...     config,
... )
True

```

### Scalar Testers

#### `ScalarEqualityTester`

Handles scalar types like `int`, `float`, and `bool`:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import ScalarEqualityTester
>>> config = EqualityConfig()
>>> tester = ScalarEqualityTester()
>>> tester.objects_are_equal(42, 42, config)
True
>>> tester.objects_are_equal(3.14, 3.14, config)
True
>>> # With tolerance
>>> config_tol = EqualityConfig(atol=0.01)
>>> tester.objects_are_equal(1.0, 1.005, config_tol)
True

```

#### `EqualEqualityTester`

Generic tester that uses Python's `==` operator:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import EqualEqualityTester
>>> config = EqualityConfig()
>>> tester = EqualEqualityTester()
>>> tester.objects_are_equal("hello", "hello", config)
True
>>> tester.objects_are_equal("hello", "world", config)
False

```

### NumPy Testers

#### `NumpyArrayEqualityTester`

Handles NumPy arrays:

```pycon

>>> import numpy as np
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import NumpyArrayEqualityTester
>>> config = EqualityConfig()
>>> tester = NumpyArrayEqualityTester()
>>> tester.objects_are_equal(
...     np.array([1, 2, 3]),
...     np.array([1, 2, 3]),
...     config,
... )
True
>>> tester.objects_are_equal(
...     np.array([1, 2, 3]),
...     np.array([1, 2, 4]),
...     config,
... )
False
>>> # With tolerance
>>> config_tol = EqualityConfig(atol=0.01)
>>> tester.objects_are_equal(
...     np.array([1.0, 2.0, 3.0]),
...     np.array([1.0, 2.0, 3.001]),
...     config_tol,
... )
True

```

#### `NumpyMaskedArrayEqualityTester`

Handles NumPy masked arrays:

```pycon

>>> import numpy as np
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import NumpyMaskedArrayEqualityTester
>>> config = EqualityConfig()
>>> tester = NumpyMaskedArrayEqualityTester()
>>> arr1 = np.ma.array([1, 2, 3], mask=[0, 0, 1])
>>> arr2 = np.ma.array([1, 2, 99], mask=[0, 0, 1])  # Masked value differs
>>> tester.objects_are_equal(arr1, arr2, config)
True

```

### PyTorch Testers

#### `TorchTensorEqualityTester`

Handles PyTorch tensors:

```pycon

>>> import torch
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import TorchTensorEqualityTester
>>> config = EqualityConfig()
>>> tester = TorchTensorEqualityTester()
>>> tester.objects_are_equal(
...     torch.tensor([1, 2, 3]),
...     torch.tensor([1, 2, 3]),
...     config,
... )
True
>>> # With tolerance
>>> config_tol = EqualityConfig(atol=0.01)
>>> tester.objects_are_equal(
...     torch.tensor([1.0, 2.0, 3.0]),
...     torch.tensor([1.0, 2.0, 3.001]),
...     config_tol,
... )
True

```

#### `TorchPackedSequenceEqualityTester`

Handles PyTorch PackedSequence objects:

```pycon

>>> import torch
>>> from torch.nn.utils.rnn import pack_padded_sequence
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import TorchPackedSequenceEqualityTester
>>> config = EqualityConfig()
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
>>> tester.objects_are_equal(ps1, ps2, config)
True

```

### pandas Testers

#### `PandasDataFrameEqualityTester`

Handles pandas DataFrames:

```pycon

>>> import pandas as pd
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import PandasDataFrameEqualityTester
>>> config = EqualityConfig()
>>> tester = PandasDataFrameEqualityTester()
>>> df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
>>> df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
>>> tester.objects_are_equal(df1, df2, config)
True

```

#### `PandasSeriesEqualityTester`

Handles pandas Series:

```pycon

>>> import pandas as pd
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.tester import PandasSeriesEqualityTester
>>> config = EqualityConfig()
>>> tester = PandasSeriesEqualityTester()
>>> s1 = pd.Series([1, 2, 3])
>>> s2 = pd.Series([1, 2, 3])
>>> tester.objects_are_equal(s1, s2, config)
True

```

### polars Testers

The package includes testers for polars DataFrames, Series, and LazyFrames:

- **`PolarsDataFrameEqualityTester`**: Compare polars DataFrames
- **`PolarsSeriesEqualityTester`**: Compare polars Series
- **`PolarsLazyFrameEqualityTester`**: Compare polars LazyFrames

### Other Library Testers

- **`JaxArrayEqualityTester`**: Compare JAX arrays
- **`XarrayDataArrayEqualityTester`**: Compare xarray DataArrays
- **`XarrayDatasetEqualityTester`**: Compare xarray Datasets
- **`XarrayVariableEqualityTester`**: Compare xarray Variables
- **`PyarrowEqualityTester`**: Compare PyArrow arrays and tables

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
...
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
>>> registry = EqualityTesterRegistry({
...     list: SequenceEqualityTester(),
...     dict: MappingEqualityTester(),
...     object: DefaultEqualityTester(),
... })
>>> config = EqualityConfig(registry=registry)
>>> registry.objects_are_equal([1, 2], [1, 2], config)
True

```

## Creating Custom Testers

To create a custom tester, inherit from `BaseEqualityTester`:

```pycon

>>> from coola.equality.tester import BaseEqualityTester
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import (
...     create_chain,
...     SameObjectHandler,
...     SameTypeHandler,
...     SameAttributeHandler,
...     TrueHandler,
... )
>>> class MyObjectEqualityTester(BaseEqualityTester):
...     """Tester for objects with a 'value' attribute."""
...
...     def __init__(self):
...         self._handler = create_chain(
...             SameObjectHandler(),
...             SameTypeHandler(),
...             SameAttributeHandler("value"),
...             TrueHandler(),
...         )
...
...     def equal(self, other: object) -> bool:
...         return type(other) is type(self)
...
...     def objects_are_equal(self, actual, expected, config: EqualityConfig) -> bool:
...         return self._handler.handle(actual, expected, config)
...

```

The tester uses a handler chain to implement the comparison logic.

## Handler Integration

Testers use handler chains from `coola.equality.handler` to implement their comparison logic:

```pycon

>>> from coola.equality.tester import HandlerEqualityTester
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import (
...     create_chain,
...     SameObjectHandler,
...     SameTypeHandler,
...     ObjectEqualHandler,
... )
>>> # Create a custom handler chain
>>> handler_chain = create_chain(
...     SameObjectHandler(),
...     SameTypeHandler(),
...     ObjectEqualHandler(),
... )
>>> # Create a tester with this handler
>>> tester = HandlerEqualityTester(handler_chain)
>>> config = EqualityConfig()
>>> tester.objects_are_equal("hello", "hello", config)
True

```

This allows you to mix and match handlers to create custom comparison logic.

## Advanced Usage

### Type Resolution with MRO

The registry uses Python's Method Resolution Order (MRO) to find testers:

```pycon

>>> from coola.equality.tester import get_default_registry
>>> from coola.equality.config import EqualityConfig
>>> from collections import OrderedDict
>>> registry = get_default_registry()
>>> config = EqualityConfig()
>>> # OrderedDict inherits from dict, so MappingEqualityTester is used
>>> # But the tester checks that both objects have the same type
>>> od = OrderedDict([("a", 1), ("b", 2)])
>>> registry.objects_are_equal(od, {"a": 1, "b": 2}, config)
False
>>> # Comparing two OrderedDicts works (same type)
>>> registry.objects_are_equal(od, OrderedDict([("a", 1), ("b", 2)]), config)
True

```

This means you can register a tester for a base class and it will work for all subclasses.

### Recursive Comparison

Testers that handle collections (sequences, mappings) recursively use the registry to compare
nested values:

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

The recursion depth is controlled by the `max_depth` parameter in `EqualityConfig`.

### Performance Considerations

The registry includes an LRU cache for type lookups to optimize performance:

```pycon

>>> from coola.equality.tester import get_default_registry
>>> registry = get_default_registry()
>>> # First lookup for a type will cache the result
>>> # Subsequent lookups for the same type will be faster
>>> # This is automatically managed internally

```

## Common Use Cases

### Testing Framework Integration

Testers are commonly used in testing frameworks:

```pycon

>>> from coola.equality.tester import get_default_registry
>>> from coola.equality.config import EqualityConfig
>>> def assert_equal(actual, expected):
...     registry = get_default_registry()
...     config = EqualityConfig(show_difference=True)
...     if not registry.objects_are_equal(actual, expected, config):
...         raise AssertionError(f"{actual} != {expected}")
...
>>> assert_equal([1, 2, 3], [1, 2, 3])  # Passes
>>> # assert_equal([1, 2, 3], [1, 2, 4])  # Would raise AssertionError

```

### Custom Type Support

Add support for your custom types:

```pycon

>>> from coola.equality.tester import (
...     register_equality_testers,
...     BaseEqualityTester,
... )
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import (
...     create_chain,
...     SameObjectHandler,
...     SameTypeHandler,
...     ObjectEqualHandler,
... )
>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
...
...     def __eq__(self, other):
...         return isinstance(other, Point) and self.x == other.x and self.y == other.y
...
>>> class PointEqualityTester(BaseEqualityTester):
...     def __init__(self):
...         self._handler = create_chain(
...             SameObjectHandler(),
...             SameTypeHandler(),
...             ObjectEqualHandler(),
...         )
...
...     def equal(self, other):
...         return type(other) is type(self)
...
...     def objects_are_equal(self, actual, expected, config):
...         return self._handler.handle(actual, expected, config)
...
>>> # Register the tester (modifies global state, so skipped in doctests)
>>> register_equality_testers({Point: PointEqualityTester()})  # doctest: +SKIP

```

### Tolerance-Based Comparisons

Use testers with tolerance configurations:

```pycon

>>> import numpy as np
>>> from coola.equality.tester import get_default_registry
>>> from coola.equality.config import EqualityConfig
>>> registry = get_default_registry()
>>> config = EqualityConfig(atol=1e-5, rtol=1e-5)
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

3. **Handler composition**: Testers use composable handler chains for flexible comparison logic

4. **Recursive comparison**: Nested structures are compared recursively using the registry

5. **Performance**: LRU caching of type lookups for efficient repeated comparisons

6. **Configurability**: `EqualityConfig` provides fine-grained control over comparison behavior

## See Also

- [`coola.equality`](equality.md): Main equality comparison functions
- [`coola.equality.handler`](equality_handler.md): Handler system for implementing comparisons
- [`coola.registry`](registry.md): General registry pattern used by the tester system

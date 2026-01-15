# Equality Handlers

:book: This page describes the `coola.equality.handler` package, which implements the handler
system for checking equality using the Chain of Responsibility pattern.

**Prerequisites:** You'll need to know a bit of Python and understand the basics of the
[coola.equality](equality.md) package.
For a refresher on Python, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.equality.handler` package provides a set of handlers that work together to check if two
objects are equal. Each handler is responsible for checking one aspect of equality (e.g., same
type, same length, same values) and delegates to the next handler in the chain if its check passes.

This follows the **Chain of Responsibility** design pattern, where each handler either:

1. Returns `False` if its specific check fails
2. Passes the request to the next handler if its check succeeds

The handlers are the building blocks used by equality testers to implement type-specific comparison
logic.

## Key Concepts

### Chain of Responsibility Pattern

Handlers are chained together to perform multiple checks in sequence:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import (
...     SameObjectHandler,
...     SameTypeHandler,
...     SameLengthHandler,
...     ObjectEqualHandler,
... )
>>> config = EqualityConfig()
>>> # Build a handler chain (returns the last handler in chain)
>>> handler = SameObjectHandler()
>>> handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(ObjectEqualHandler())
ObjectEqualHandler()
>>> # Use the chain
>>> handler.handle([1, 2, 3], [1, 2, 3], config)
True
>>> handler.handle([1, 2, 3], [1, 2, 4], config)
False

```

Each handler in the chain performs a specific check and only advances to the next handler if its
check passes.

### Base Handler Class

All handlers inherit from `BaseEqualityHandler`, which provides the interface and chain management:

```pycon

>>> from coola.equality.handler import BaseEqualityHandler, SameTypeHandler
>>> handler = SameTypeHandler()
>>> isinstance(handler, BaseEqualityHandler)
True

```

## Common Handlers

### Structural Handlers

These handlers check structural properties of objects:

#### `SameObjectHandler`

Checks if both objects are the same object (identity check using `is`):

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import SameObjectHandler, FalseHandler
>>> config = EqualityConfig()
>>> handler = SameObjectHandler(FalseHandler())
>>> # Same object
>>> obj = [1, 2, 3]
>>> handler.handle(obj, obj, config)
True
>>> # Different objects
>>> handler.handle([1, 2, 3], [1, 2, 3], config)
False

```

This is an optimization - if two objects are the same object in memory, they must be equal.

#### `SameTypeHandler`

Checks if both objects have the same type:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import SameTypeHandler, TrueHandler, FalseHandler
>>> config = EqualityConfig()
>>> handler = SameTypeHandler(TrueHandler())
>>> handler.handle([1, 2, 3], [1, 2, 3], config)
True
>>> handler.handle([1, 2, 3], (1, 2, 3), config)
False

```

This ensures that lists are only compared with lists, dicts with dicts, etc.

#### `SameLengthHandler`

Checks if both objects have the same length (for sized objects):

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import SameLengthHandler, TrueHandler
>>> config = EqualityConfig()
>>> handler = SameLengthHandler(TrueHandler())
>>> handler.handle([1, 2, 3], [4, 5, 6], config)
True
>>> handler.handle([1, 2, 3], [4, 5], config)
False

```

#### `SameShapeHandler`

Checks if both objects have the same shape (for arrays/tensors):

```pycon

>>> import numpy as np
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import SameShapeHandler, TrueHandler
>>> config = EqualityConfig()
>>> handler = SameShapeHandler(TrueHandler())
>>> handler.handle(np.ones((2, 3)), np.zeros((2, 3)), config)
True
>>> handler.handle(np.ones((2, 3)), np.zeros((3, 2)), config)
False

```

#### `SameDTypeHandler`

Checks if both objects have the same data type (for arrays/tensors):

```pycon

>>> import numpy as np
>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import SameDTypeHandler, TrueHandler
>>> config = EqualityConfig()
>>> handler = SameDTypeHandler(TrueHandler())
>>> handler.handle(np.array([1, 2, 3], dtype=int), np.array([1, 2, 3], dtype=int), config)
True
>>> handler.handle(np.array([1, 2, 3], dtype=int), np.array([1, 2, 3], dtype=float), config)
False

```

### Value Comparison Handlers

These handlers compare the actual values of objects:

#### `ObjectEqualHandler`

Uses Python's `==` operator to check equality:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import ObjectEqualHandler
>>> config = EqualityConfig()
>>> handler = ObjectEqualHandler()
>>> handler.handle("abc", "abc", config)
True
>>> handler.handle("abc", "def", config)
False

```

#### `EqualHandler`

Recursively checks if objects are equal using the equality registry:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import EqualHandler
>>> class MyList(list):
...     def equal(self, other: object) -> bool:
...         return self == other
...
>>> config = EqualityConfig()
>>> handler = EqualHandler()
>>> handler.handle(MyList([1, 2, 3]), MyList([1, 2, 3]), config)
True
>>> handler.handle(MyList([1, 2, 3]), MyList([1, 2, 4]), config)
False

```

This handler is used to compare nested values recursively.

#### `ScalarEqualHandler`

Compares scalar values (int, float, bool) with optional tolerance:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import ScalarEqualHandler
>>> config = EqualityConfig(atol=0.1)
>>> handler = ScalarEqualHandler()
>>> handler.handle(1.0, 1.05, config)
True
>>> config2 = EqualityConfig(atol=0.01)
>>> handler.handle(1.0, 1.05, config2)
False

```

#### `NanEqualHandler`

Special handler for comparing NaN values:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import NanEqualHandler
>>> config = EqualityConfig(equal_nan=True)
>>> handler = NanEqualHandler(FalseHandler())
>>> handler.handle(float("nan"), float("nan"), config)
True
>>> config2 = EqualityConfig(equal_nan=False)
>>> handler.handle(float("nan"), float("nan"), config2)
False

```

### Collection Handlers

These handlers work with collections like lists, tuples, and dictionaries:

#### `SequenceSameValuesHandler`

Compares sequence values element by element:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import SequenceSameValuesHandler, create_chain
>>> config = EqualityConfig()
>>> handler = create_chain(SequenceSameValuesHandler(), TrueHandler())
>>> handler.handle([1, 2, 3], [1, 2, 3], config)
True
>>> handler.handle([1, 2, 3], [1, 2, 4], config)
False

```

#### `MappingSameKeysHandler`

Checks if two mappings have the same keys:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import MappingSameKeysHandler, TrueHandler, create_chain
>>> config = EqualityConfig()
>>> handler = create_chain(MappingSameKeysHandler(), TrueHandler())
>>> handler.handle({"a": 1, "b": 2}, {"a": 3, "b": 4}, config)
True
>>> handler.handle({"a": 1, "b": 2}, {"a": 3, "c": 4}, config)
False

```

#### `MappingSameValuesHandler`

Compares mapping values for the same keys:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import MappingSameValuesHandler, TrueHandler, create_chain
>>> config = EqualityConfig()
>>> handler = create_chain(MappingSameValuesHandler(), TrueHandler())
>>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 2}, config)
True
>>> handler.handle({"a": 1, "b": 2}, {"a": 1, "b": 3}, config)
False

```

### Type-Specific Handlers

The package includes specialized handlers for specific types:

- **`NumpyArrayEqualHandler`**: Compare NumPy arrays
- **`TorchTensorEqualHandler`**: Compare PyTorch tensors
- **`TorchTensorSameDeviceHandler`**: Check if tensors are on the same device
- **`PandasDataFrameEqualHandler`**: Compare pandas DataFrames
- **`PandasSeriesEqualHandler`**: Compare pandas Series
- **`PolarsDataFrameEqualHandler`**: Compare polars DataFrames
- **`PolarsSeriesEqualHandler`**: Compare polars Series
- **`JaxArrayEqualHandler`**: Compare JAX arrays
- **`PyarrowEqualHandler`**: Compare PyArrow arrays and tables

### Terminal Handlers

These handlers end the chain:

#### `TrueHandler`

Always returns `True` (used when all checks pass):

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import TrueHandler
>>> config = EqualityConfig()
>>> handler = TrueHandler()
>>> handler.handle("anything", "else", config)
True

```

#### `FalseHandler`

Always returns `False` (used as a fallback):

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import FalseHandler
>>> config = EqualityConfig()
>>> handler = FalseHandler()
>>> handler.handle("same", "same", config)
False

```

## Building Handler Chains

### Manual Chain Building

You can build chains manually using the `chain()` method:

```pycon

>>> from coola.equality.handler import (
...     SameObjectHandler,
...     SameTypeHandler,
...     SameLengthHandler,
...     ObjectEqualHandler,
... )
>>> handler = SameObjectHandler()
>>> handler.chain(SameTypeHandler()).chain(SameLengthHandler()).chain(ObjectEqualHandler())
ObjectEqualHandler()
>>> print(handler.visualize_chain())
(0): SameObjectHandler()
(1): SameTypeHandler()
(2): SameLengthHandler()
(3): ObjectEqualHandler()

```

### Using `chain_all()`

Build chains with multiple handlers at once:

```pycon

>>> from coola.equality.handler import (
...     SameObjectHandler,
...     SameTypeHandler,
...     SameLengthHandler,
...     ObjectEqualHandler,
... )
>>> handler = SameObjectHandler()
>>> handler.chain_all(SameTypeHandler(), SameLengthHandler(), ObjectEqualHandler())
ObjectEqualHandler()
>>> handler.get_chain_length()
4

```

### Using `create_chain()`

The utility function `create_chain()` creates a chain from multiple handlers:

```pycon

>>> from coola.equality.handler import (
...     create_chain,
...     SameObjectHandler,
...     SameTypeHandler,
...     ObjectEqualHandler,
... )
>>> handler = create_chain(SameObjectHandler(), SameTypeHandler(), ObjectEqualHandler())
>>> print(handler.visualize_chain())
(0): SameObjectHandler()
(1): SameTypeHandler()
(2): ObjectEqualHandler()

```

## Advanced Usage

### Custom Handlers

You can create custom handlers by inheriting from `BaseEqualityHandler`:

```pycon

>>> from coola.equality.handler import BaseEqualityHandler
>>> from coola.equality.config import EqualityConfig
>>> class CustomHandler(BaseEqualityHandler):
...     def equal(self, other: object) -> bool:
...         return type(other) is type(self)
...     def handle(self, actual: object, expected: object, config: EqualityConfig) -> bool:
...         # Custom comparison logic
...         if self._meets_condition(actual, expected):
...             return self._handle_next(actual, expected, config)
...         return False
...     def _meets_condition(self, actual: object, expected: object) -> bool:
...         # Implement your condition
...         return True
...

```

### Inspecting Handler Chains

You can visualize and inspect handler chains:

```pycon

>>> from coola.equality.handler import (
...     create_chain,
...     SameObjectHandler,
...     SameTypeHandler,
...     TrueHandler,
... )
>>> handler = create_chain(SameObjectHandler(), SameTypeHandler(), TrueHandler())
>>> # Get chain length
>>> handler.get_chain_length()
3
>>> # Visualize the chain
>>> print(handler.visualize_chain())
(0): SameObjectHandler()
(1): SameTypeHandler()
(2): TrueHandler()

```

### Handler Equality

Handlers can be compared for equality:

```pycon

>>> from coola.equality.handler import (
...     SameTypeHandler,
...     SameObjectHandler,
...     handlers_are_equal,
... )
>>> handler1 = SameTypeHandler()
>>> handler2 = SameTypeHandler()
>>> handler3 = SameObjectHandler()
>>> handler1.equal(handler2)
True
>>> handler1.equal(handler3)
False
>>> handlers_are_equal(handler1, handler2)
True

```

## Design Patterns

### Typical Handler Chain Structure

Most equality testers use a similar structure:

1. **Identity check**: `SameObjectHandler` (optimization)
2. **Type check**: `SameTypeHandler` (ensure compatible types)
3. **Structural checks**: `SameLengthHandler`, `SameShapeHandler`, etc.
4. **Value comparison**: Type-specific handlers
5. **Terminal handler**: `TrueHandler` (all checks passed)

Example for sequences:

```pycon

>>> from coola.equality.handler import (
...     create_chain,
...     SameObjectHandler,
...     SameTypeHandler,
...     SameLengthHandler,
...     SequenceSameValuesHandler,
...     TrueHandler,
... )
>>> handler = create_chain(
...     SameObjectHandler(),
...     SameTypeHandler(),
...     SameLengthHandler(),
...     SequenceSameValuesHandler(),
...     TrueHandler(),
... )

```

### Recursion Control

The `EqualityConfig` object passed through the chain tracks recursion depth to prevent infinite
loops in circular structures:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> config = EqualityConfig(max_depth=100)
>>> config.depth
0
>>> config.increment_depth()
>>> config.depth
1

```

This is automatically managed by the handlers that perform recursive comparisons.

## Common Use Cases

### Implementing Custom Type Comparisons

When adding support for a new type, you create a handler chain specific to that type:

```pycon

>>> from coola.equality.handler import (
...     create_chain,
...     SameObjectHandler,
...     SameTypeHandler,
...     SameAttributeHandler,
...     TrueHandler,
... )
>>> # Handler chain for comparing objects with 'value' attribute
>>> handler = create_chain(
...     SameObjectHandler(),
...     SameTypeHandler(),
...     SameAttributeHandler("value"),
...     TrueHandler(),
... )

```

### Optimizing Comparisons

The handler chain allows short-circuiting - if any handler returns `False`, the remaining handlers
are not executed:

```pycon

>>> from coola.equality.config import EqualityConfig
>>> from coola.equality.handler import (
...     create_chain,
...     SameTypeHandler,
...     TrueHandler,
... )
>>> config = EqualityConfig()
>>> handler = create_chain(SameTypeHandler(), TrueHandler())
>>> # Type check fails, TrueHandler is never called
>>> handler.handle([1, 2], (1, 2), config)
False

```

## See Also

- [`coola.equality`](equality.md): Main equality comparison functions
- [`coola.equality.tester`](equality_tester.md): Equality tester registry system
- [Chain of Responsibility Pattern](https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern): Design pattern used by handlers

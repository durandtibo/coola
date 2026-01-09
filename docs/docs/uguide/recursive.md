# Recursive Data Transformation

:book: This page describes the `coola.recursive` package, which provides utilities for recursively
applying transformations to nested data structures using a Depth-First Search (DFS) pattern.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.recursive` package allows you to apply a function to all leaf values in nested data
structures (lists, dicts, tuples, sets, etc.) while preserving the original structure. It provides:

1. Memory-efficient generator-based traversal
2. Clean separation between transformation logic and type dispatch
3. Easy extensibility via registry pattern
4. Support for custom types

## Basic Usage

### Transforming Nested Data

The main function is `recursive_apply`, which recursively applies a function to all items in nested
data:

```pycon

>>> from coola.recursive import recursive_apply
>>> recursive_apply({"a": 1, "b": "abc"}, str)
{'a': '1', 'b': 'abc'}
>>> recursive_apply([1, [2, 3], {"x": 4}], lambda x: x * 2)
[2, [4, 6], {'x': 8}]

```

The function traverses the data structure and applies the transformation function to each leaf value
(i.e., non-container values like numbers and strings).

### More Examples

You can use `recursive_apply` with different data structures:

**Nested lists and tuples:**

```pycon

>>> from coola.recursive import recursive_apply
>>> recursive_apply([1, 2, [3, 4, [5, 6]]], lambda x: x + 10)
[11, 12, [13, 14, [15, 16]]]
>>> recursive_apply((1, (2, 3)), str)
('1', ('2', '3'))

```

**Nested dictionaries:**

```pycon

>>> from coola.recursive import recursive_apply
>>> data = {"level1": {"level2": {"level3": 42}}}
>>> recursive_apply(data, lambda x: x * 2)
{'level1': {'level2': {'level3': 84}}}

```

**Sets:**

```pycon

>>> from coola.recursive import recursive_apply
>>> recursive_apply({1, 2, 3}, lambda x: x ** 2)
{1, 4, 9}

```

**Mixed nested structures:**

```pycon

>>> from coola.recursive import recursive_apply
>>> data = {
...     "list": [1, 2, 3],
...     "dict": {"a": 4, "b": 5},
...     "set": {6, 7},
...     "value": 8,
... }
>>> recursive_apply(data, lambda x: x + 100)
{'list': [101, 102, 103], 'dict': {'a': 104, 'b': 105}, 'set': {106, 107}, 'value': 108}

```

## Advanced Usage

### Custom Transformers

For more control over how specific types are transformed, you can create custom transformers by
extending `BaseTransformer`:

```pycon

>>> from coola.recursive import BaseTransformer, register_transformers
>>> class MyType:
...     def __init__(self, value):
...         self.value = value
...     def __repr__(self):
...         return f"MyType({self.value})"
...
>>> class MyTransformer(BaseTransformer):
...     def transform(self, data, func, registry):
...         return MyType(func(data.value))
...
>>> register_transformers({MyType: MyTransformer()})

```

### Using Custom Registry

You can create and use a custom registry for more control:

```pycon

>>> from coola.recursive import TransformerRegistry, recursive_apply
>>> from coola.recursive import SequenceTransformer, DefaultTransformer
>>> registry = TransformerRegistry()
>>> registry.register(list, SequenceTransformer())
>>> registry.register(object, DefaultTransformer())
>>> recursive_apply([1, 2, 3], lambda x: x * 10, registry=registry)
[10, 20, 30]

```

## Available Transformers

The `coola.recursive` package provides several built-in transformers:

- **`DefaultTransformer`**: For scalar/leaf values (no recursion)
- **`SequenceTransformer`**: For sequences (list, tuple) - recursive transformation preserving order
- **`MappingTransformer`**: For mappings (dict) - recursive transformation of keys and values
- **`SetTransformer`**: For sets - recursive transformation without order
- **`ConditionalTransformer`**: For conditional transformation based on predicates

## Registry System

### Getting the Default Registry

The package maintains a singleton default registry with transformers for common Python types:

```pycon

>>> from coola.recursive import get_default_registry
>>> registry = get_default_registry()
>>> registry.transform([1, 2, 3], str)
['1', '2', '3']
>>> registry.transform({"a": 1, "b": 2}, lambda x: x * 10)
{'a': 10, 'b': 20}

```

The default registry includes transformers for:
- Scalar types: `int`, `float`, `complex`, `bool`, `str`
- Sequences: `list`, `tuple`, `Sequence` (ABC)
- Sets: `set`, `frozenset`
- Mappings: `dict`, `Mapping` (ABC)

### Registering Custom Types

You can extend the default registry to support custom types:

```pycon

>>> from coola.recursive import register_transformers, BaseTransformer
>>> class Point:
...     def __init__(self, x, y):
...         self.x = x
...         self.y = y
...
>>> class PointTransformer(BaseTransformer):
...     def transform(self, data, func, registry):
...         return Point(func(data.x), func(data.y))
...
>>> register_transformers({Point: PointTransformer()})

```

## Design Principles

The `coola.recursive` package design is inspired by the DFS pattern and provides:

1. **Memory-efficient traversal**: Uses generators to avoid loading entire structures into memory
2. **Type dispatch**: Automatically selects the right transformer based on data type
3. **No state threading**: Clean API without passing state through recursion
4. **Extensibility**: Easy to add support for new types via the registry pattern

## Common Use Cases

### Data Type Conversion

Convert all numeric values in a nested structure to strings:

```pycon

>>> from coola.recursive import recursive_apply
>>> data = {"metrics": [1.5, 2.3, 3.7], "count": 42}
>>> recursive_apply(data, str)
{'metrics': ['1.5', '2.3', '3.7'], 'count': '42'}

```

### Value Scaling

Scale all numeric values in a configuration:

```pycon

>>> from coola.recursive import recursive_apply
>>> config = {
...     "learning_rate": 0.001,
...     "layers": [64, 128, 256],
...     "dropout": 0.5,
... }
>>> recursive_apply(config, lambda x: x * 10 if isinstance(x, (int, float)) else x)
{'learning_rate': 0.01, 'layers': [640, 1280, 2560], 'dropout': 5.0}

```

### Data Normalization

Normalize all values to a specific range:

```pycon

>>> from coola.recursive import recursive_apply
>>> data = {"a": 100, "b": [200, 300], "c": {"d": 400}}
>>> recursive_apply(data, lambda x: x / 100)
{'a': 1.0, 'b': [2.0, 3.0], 'c': {'d': 4.0}}

```

## See Also

- [`coola.iterator`](iterator.md): For iterating over nested data without transformation
- [`coola.registry`](registry.md): For understanding the registry pattern used internally

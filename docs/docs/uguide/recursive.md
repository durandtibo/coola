# Recursive Data Transformation

:book: This page describes the `coola.recursive` package, which provides utilities for recursively
applying transformations to nested data structures using a Depth-First Search (DFS) pattern.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.recursive` package allows you to apply a function to all leaf values in nested data
structures (lists, dicts, tuples, sets, etc.) while preserving the original structure. It provides:

1. Clean separation between transformation logic and type dispatch
2. Easy extensibility via registry pattern
3. Support for custom types

## Properties

| Property                 | Statement                                                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Leaves only              | `func` is called on the leaves, once per occurrence, never on a container.                                                                                                          |
| Structure-preserving     | A container is rebuilt from the transformed children with its original type. Keys are never transformed.                                                                            |
| Type dispatch            | The transformer is chosen from the type of the data, along the Method Resolution Order, so the most specific type wins.                                                             |
| Registry-extensible      | A new type is supported by registering a transformer, without changing the existing ones.                                                                                           |
| Deterministic order      | The leaves are visited depth first, left to right. The elements of a set are visited in the iteration order of the set, which is not specified.                                     |
| Pure                     | The input is never modified. The output shares no container with the input (leaves can be shared).                                                                                  |
| Cycle-safe               | A container that contains itself raises a `ValueError`.                                                                                                                             |
| Errors propagate         | An exception raised by `func` propagates unchanged. A container that cannot be rebuilt raises a `TypeError`.                                                                        |

Not guaranteed:

- **Laziness.** The output is built whole: the traversal does not use generators and does not
  reduce the memory usage.
- **Shared references.** A container reachable by several paths is transformed once per path, so
  `func` is called once per occurrence and the cost is proportional to the number of paths.
- **Order of the calls inside a set**, and the result of a `func` that is not deterministic.
- **Depth.** The recursion limit of Python applies: a very deep structure raises a
  `RecursionError`.

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

````

**Mixed nested structures:**

```pycon
>>> from coola.recursive import recursive_apply
>>> data = {
...     "list": [1, 2, 3],
...     "dict": {"a": 4, "b": 5},
...     "value": 8,
... }
>>> recursive_apply(data, lambda x: x + 100)
{'list': [101, 102, 103], 'dict': {'a': 104, 'b': 105}, 'value': 108}

````

## Advanced Usage

### Custom Transformers

For more control over how specific types are transformed, you can create custom transformers by
extending `BaseTransformer`:

```pycon
>>> from coola.recursive import BaseTransformer, TransformerRegistry
>>> class MyType:
...     def __init__(self, value):
...         self.value = value
...
...     def __repr__(self):
...         return f"MyType({self.value})"
>>> class MyTransformer(BaseTransformer):
...     def transform(self, data, func, registry):
...         return MyType(func(data.value))
>>> registry = TransformerRegistry()
>>> registry.register(MyType, MyTransformer())

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

### Dataclasses

By default, a dataclass instance is a leaf: `recursive_apply` calls the function on the whole
object, while `objects_are_equal`, `hash_object` and `summarize` enter the dataclasses. Call
`enter_dataclasses()` to transform the fields instead. The instance is rebuilt with
`dataclasses.replace` (`DataclassTransformer`), so frozen dataclasses are supported and the input
is not modified. The fields with `init=False` are recomputed by `__init__`/`__post_init__`:

```pycon
>>> from dataclasses import dataclass
>>> from coola.recursive import (
...     DefaultTransformer,
...     SequenceTransformer,
...     TransformerRegistry,
...     enter_dataclasses,
...     recursive_apply,
... )
>>> @dataclass
... class Config:
...     lr: float
...     layers: list
>>> registry = TransformerRegistry({object: DefaultTransformer(), list: SequenceTransformer()})
>>> enter_dataclasses(registry=registry)
>>> recursive_apply(Config(0.1, [1, 2]), str, registry=registry)
Config(lr='0.1', layers=['1', '2'])

```

Without `registry`, `enter_dataclasses()` changes the default registry for the whole process. A
transformer registered for the type of the dataclass has priority.

## Available Transformers

The `coola.recursive` package provides several built-in transformers:

- **`DefaultTransformer`**: For scalar/leaf values (no recursion)
- **`SequenceTransformer`**: For sequences (list, tuple) - recursive transformation preserving order
- **`MappingTransformer`**: For mappings (dict) - recursive transformation of the values (the keys are kept)
- **`SetTransformer`**: For sets - recursive transformation without order
- **`IdentityTransformer`**: For leaf values that must be kept as they are: the function is never
  applied
- **`KeyFilterTransformer`**: For mappings: drops the entries whose key satisfies a predicate and
  transforms the remaining values

```pycon
>>> from coola.recursive import (
...     DefaultTransformer,
...     KeyFilterTransformer,
...     TransformerRegistry,
... )
>>> registry = TransformerRegistry(
...     {
...         object: DefaultTransformer(),
...         dict: KeyFilterTransformer(key_predicate=lambda key: key == "secret"),
...     }
... )
>>> registry.transform({"keep": 1, "secret": 2, "nested": {"secret": 3, "x": 4}}, str)
{'keep': '1', 'nested': {'x': '4'}}

```

## Registry System

### Getting the Default Registry

!!! tip "Unambiguous alias"

    `get_default_registry` exists in several `coola` subpackages with different return types.
    `get_default_transformer_registry` is an alias of the same function with a more explicit name, which avoids
    import collisions.

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
>>> class PointTransformer(BaseTransformer):
...     def transform(self, data, func, registry):
...         return Point(func(data.x), func(data.y))
>>> register_transformers({Point: PointTransformer()})  # doctest: +SKIP

```

## Design Principles

The `coola.recursive` package design is inspired by the DFS pattern and provides:

1. **Type dispatch**: Automatically selects the right transformer based on data type
2. **Extensibility**: Easy to add support for new types via the registry pattern

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

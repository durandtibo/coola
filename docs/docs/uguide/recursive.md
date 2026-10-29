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

| Property             | Statement                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Leaves only          | `func` is called on the leaves, once per occurrence, never on a container.                                                                                                     |
| Structure-preserving | A container is rebuilt from the transformed children with its original type. Keys are never transformed. Dataclasses are entered, like in `equality`, `hashing` and `summary`. |
| Type dispatch        | The transformer is chosen from the type of the data, along the Method Resolution Order, so the most specific type wins.                                                        |
| Registry-extensible  | A new type is supported by registering a transformer, without changing the existing ones.                                                                                      |
| Deterministic order  | The leaves are visited depth first, left to right. The elements of a set are visited in the iteration order of the set, which is not specified.                                |
| Pure                 | The input is never modified. The output shares no container with the input (leaves can be shared).                                                                             |
| Any depth            | No `RecursionError`, whatever the nesting: the traversal uses an explicit stack. The depth is bounded by memory only.                                                          |
| Cycle-safe           | A container that contains itself raises a `ValueError`.                                                                                                                        |
| Errors propagate     | An exception raised by `func` propagates unchanged. A container that cannot be rebuilt raises a `TypeError`.                                                                   |

Not guaranteed:

- **Laziness.** The output is built whole: the traversal does not use generators and does not
  reduce the memory usage.
- **Shared references.** A container reachable by several paths is transformed once per path, so
  `func` is called once per occurrence and the cost is proportional to the number of paths.
- **Order of the calls inside a set**, and the result of a `func` that is not deterministic.

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
extending `BaseTransformer`. A transformer only describes the shape of a type: `expand(data)`
returns `None` for a leaf, or the children and the function that rebuilds the value from the
transformed children. The engine applies the function to the leaves and detects the cycles:

```pycon
>>> from coola.recursive import (
...     BaseTransformer,
...     DefaultTransformer,
...     SequenceTransformer,
...     TransformerRegistry,
...     recursive_apply,
... )
>>> class MyType:
...     def __init__(self, value):
...         self.value = value
...
...     def __repr__(self):
...         return f"MyType({self.value})"
>>> class MyTransformer(BaseTransformer):
...     def expand(self, data):
...         return [data.value], lambda items: MyType(items[0])
>>> registry = TransformerRegistry(
...     {object: DefaultTransformer(), list: SequenceTransformer(), MyType: MyTransformer()}
... )
>>> recursive_apply([MyType(1), MyType(2)], lambda x: x * 10, registry=registry)
[MyType(10), MyType(20)]

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

A dataclass instance is entered, like `objects_are_equal`, `hash_object` and `summarize` do: the
function is applied to the fields, not to the instance. The instance is rebuilt (`DataclassTransformer`)
from a shallow copy in which every field is set with `object.__setattr__`, so frozen and slotted
dataclasses are supported, the fields with `init=False` are transformed, the input is not modified
and no user code (`__init__`, `__post_init__`) runs during the traversal:

```pycon
>>> from dataclasses import dataclass
>>> from coola.recursive import recursive_apply
>>> @dataclass
... class Config:
...     lr: float
...     layers: list
>>> recursive_apply(Config(0.1, [1, 2]), str)
Config(lr='0.1', layers=['1', '2'])

```

`__post_init__` is not run again, so a field that `__post_init__` derives from another field is
transformed like the other fields and is not recomputed. A transformer registered for the type of
the dataclass has priority over the dataclass transformer, so a dataclass is kept as a leaf by
registering `DefaultTransformer()` for its type:

```pycon
>>> from coola.recursive import DefaultTransformer, get_default_registry
>>> registry = get_default_registry().copy()
>>> registry.register(Config, DefaultTransformer())
>>> recursive_apply(Config(0.1, [1, 2]), repr, registry=registry)
'Config(lr=0.1, layers=[1, 2])'

```

## Available Transformers

The `coola.recursive` package provides several built-in transformers:

- **`DefaultTransformer`**: For scalar/leaf values (no children)
- **`SequenceTransformer`**: For sequences (list, tuple) - recursive transformation preserving order
- **`MappingTransformer`**: For mappings (dict) - recursive transformation of the values (the keys are kept)
- **`SetTransformer`**: For sets - recursive transformation without order
- **`IdentityTransformer`**: For leaf values that must be kept as they are: the function is never
  applied

To drop or rename the keys, see `remove_keys_if` and `add_prefix_suffix_to_keys` of
[`coola.nested`](nested.md): `recursive` itself never transforms a key.

### Rebuilding Containers

A container is rebuilt with the constructor of its type, so the subclasses keep their type. The
`coola.recursive.rebuild` module contains the helpers (`rebuild_sequence`, `rebuild_mapping` and
`rebuild_set`), which a custom container transformer can reuse. They handle the named tuples, the
`deque` (`maxlen`) and the `defaultdict` (factory). A `collections.ChainMap` is rebuilt with a
single flat map: its parents are not part of the structure.

If the constructor of a subclass differs from the one of its base type, a `TypeError` that names
the type is raised: register a transformer for this type.

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
...     def expand(self, data):
...         return [data.x, data.y], lambda items: Point(*items)
>>> register_transformers({Point: PointTransformer()})  # doctest: +SKIP

```

## What `coola.recursive` does not do

- **It does not transform the keys of a mapping.** To rename or remove keys, use
  [`coola.nested`](nested.md) (`add_prefix_suffix_to_keys`, `remove_keys_if`).
- **It does not enter pydantic models, attrs classes or plain objects with `__dict__`.** They are
  leaves, like NumPy arrays, tensors and data frames: `func` receives the whole object, and
  applying a function element by element is the job of `func`. Register a transformer to enter a
  type.
- **It does not report the path of a failure.** An exception raised by `func` propagates unchanged,
  without the position of the leaf in the structure.
- **It does not memoize shared containers.** A container reachable by several paths is transformed
  once per path, so an impure `func` (a counter, a random draw, a call that logs) is called once per
  occurrence and the output has no aliasing.
- **A plugin that implements `transform(data, func, registry)`** has to be ported to `expand`; the
  entry-point group is `coola.recursive.transformers`.

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

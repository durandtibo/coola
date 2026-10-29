# Summarizing Nested Data

:book: This page describes the `coola.summary` package, which provides utilities for creating
human-readable text summaries of complex and nested data structures.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.summary` package generates the text summary of a Python object, with special support for
nested data structures. Unlike Python's built-in `repr()` or `str()`, it offers:

- **Bounded output**: limit the depth, the number of items and the length of the texts
- **Bounded work**: the cost depends on the size of the summary, not on the size of the data
- **Type-specific formatting**: each type is described by a summarizer chosen from its type
- **Extensibility**: register a summarizer for your own types
- **Robustness**: a cycle, a very deep structure or an object whose `__str__` raises does not fail

This is particularly useful for debugging, logging, and displaying complex data in a readable
format without being overwhelmed by deeply nested or large structures.

## Properties

A summary is a short text that a person reads in a log, a notebook or an error message, to know what
a (possibly huge, nested, partly unprintable) object contains. These are the rules of the package:

| Property                   | Statement                                                                                                                                                                                                                                                                                                                                                                         |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bounded output             | A node shows at most `max_items` children (items, fields, columns), a text is cut at `max_characters`, and the expansion stops at `max_depth`. What is not shown is announced (`... (+N more)`), never silently dropped.                                                                                                                                                          |
| Bounded work               | The cost depends on the size of the output, not on the size of the data: a collection is never iterated beyond `max_items`, a node below `max_depth` is never expanded, and a sub-structure that is not shown is not converted to a string. Listed exceptions: `len()`, sets (the smallest items are selected in one pass), and the `repr` of an object that has no bounded form. |
| Display depth              | `max_depth` is the number of levels that are expanded. Deeper nodes are shown on **one line** (the compact form). It never raises, unlike the `max_depth` of `objects_are_equal` (a limit).                                                                                                                                                                                       |
| Total                      | Summarizing never fails because of the data: a `__str__`, `__repr__`, `__len__` or attribute that raises is replaced by a placeholder that names the exception type. Invalid options raise `ValueError` before any work.                                                                                                                                                          |
| Cycle-safe                 | A container that contains itself is shown as `<cycle>` where it comes back, whatever the summarizer registered for it.                                                                                                                                                                                                                                                            |
| Any depth                  | No `RecursionError`, whatever `max_depth` or the caller's stack: the engine uses an explicit stack. The size of the output is bounded by `max_depth`, `max_items` and `max_characters`.                                                                                                                                                                                           |
| Type dispatch              | The summarizer is chosen from `type(data)` along the Method Resolution Order, so the most specific registered type wins. A summarizer is always found: `object` is registered by the default registry.                                                                                                                                                                            |
| Registry-extensible        | A new type is supported by registering one summarizer, without changing another one and without subclassing a base class (a protocol is enough).                                                                                                                                                                                                                                  |
| Metadata first             | An array, a tensor, a data frame, a table shows its metadata (type, shape, dtype, columns), not its values. `show_data=True` shows a bounded `repr` instead. A lazy frame is never collected.                                                                                                                                                                                     |
| Secrets stay hidden        | The fields that are marked as secret (`SecretStr`, `SecretBytes`) or excluded from the `repr` (`field(repr=False)`) are not shown.                                                                                                                                                                                                                                                |
| Iterators are not consumed | A generator or an iterator is a leaf: its `str` is shown and no item is taken from it.                                                                                                                                                                                                                                                                                            |
| Deterministic              | The same value gives the same text across calls and processes: the order of a list, a tuple or a dict is the order of the data, and a set shows its smallest items in order (it does not depend on `PYTHONHASHSEED`). Only the address in a placeholder or in an unknown `repr` can change.                                                                                       |
| Pure                       | The data is never modified. Summarizers keep no state: a registry, a config and a summarizer can be shared by threads and calls.                                                                                                                                                                                                                                                  |
| Consistent with siblings   | Dataclasses are entered (like `objects_are_equal`, `hash_object`, `recursive_apply`); iterators are leaves (like `dfs_iterate`); options are keyword-only.                                                                                                                                                                                                                        |
| For humans                 | The output is meant to be read, not parsed: its exact format can change in a release.                                                                                                                                                                                                                                                                                             |

Not guaranteed:

- **Shared references.** A container reachable by several paths is shown once per path.
- **The time of a foreign `str` or `repr`.** The package bounds what it iterates, not what a user
  class does inside `__str__` or `__repr__`.
- **The bound on a chain.** Each level is indented, so a chain of `n` levels costs `O(n²)`
  characters. `max_depth` is what bounds the output: a user who asks for a huge depth gets a huge
  text.
- **A stable format** between releases (see "For humans").

What the package does not do:

- **Plain objects with `__dict__`, `attrs` classes and `__slots__`** are leaves. Register a
  summarizer to show their attributes.
- **A machine-readable form** (JSON, a tree of nodes): the package produces text for humans.
- **Width-aware or colored output**, a template for the format, and options per type. To show 3 items
  for a list and 10 for a dict, register your own summarizer.
- **Summaries of the elements of an array** (min, max, mean, number of NaN): the default is the shape
  and the dtype, which is cheap and needs no copy and no device transfer. A metadata summarizer can
  compute more for the types that want it.

## Basic Usage

### The `summarize()` Function

The simplest way to create a summary is using the `summarize()` function:

```pycon
>>> from coola.summary import summarize
>>> print(summarize({"a": 1, "b": "abc"}))
<class 'dict'> (length=2)
  (a): 1
  (b): abc

```

All the options are keyword arguments of `summarize()`:

| Option           | Default | Meaning                                                                                 |
| ---------------- | ------- | --------------------------------------------------------------------------------------- |
| `max_depth`      | `1`     | The number of levels that are expanded. The deeper nodes are shown on one line.         |
| `max_items`      | `5`     | The maximum number of items, fields or columns shown for a node. `None` shows them all. |
| `max_characters` | `1000`  | The maximum number of characters of a text. `None` does not cut the texts.              |
| `show_data`      | `False` | Show a bounded `repr` of the values of arrays, tensors and frames, not their metadata.  |
| `registry`       | `None`  | The registry that gives the summarizer of each type (the default registry if `None`).   |

The options are validated before any work: a negative number raises a `ValueError`, and a value of
the wrong type (a `bool` for `max_depth`, ...) raises a `TypeError`.

### Simple Values

For simple scalar values, the summary shows the type and the value:

```pycon
>>> from coola.summary import summarize
>>> print(summarize(42))
<class 'int'> 42
>>> print(summarize("hello"))
<class 'str'> hello
>>> print(summarize(True))
<class 'bool'> True

```

### Lists, Tuples and Sets

Sequences are displayed with their type, length, and indexed items:

```pycon
>>> from coola.summary import summarize
>>> print(summarize([1, 2, 3]))
<class 'list'> (length=3)
  (0): 1
  (1): 2
  (2): 3
>>> print(summarize((10, 20)))
<class 'tuple'> (length=2)
  (0): 10
  (1): 20

```

The items of a named tuple are labeled with their field name, like the fields of a dataclass.

A set shows its `max_items` smallest items, in order, so the summary is the same from one run to
the next (a set of strings is not iterated in the same order with a different `PYTHONHASHSEED`).
The items that cannot be compared (for example an `int` and a `str`) are shown in iteration order:

```pycon
>>> from coola.summary import summarize
>>> print(summarize({3, 1, 2}))
<class 'set'> (length=3)
  (0): 1
  (1): 2
  (2): 3

```

### Dictionaries

Dictionaries show their type, length, and key-value pairs, in the order of the dictionary:

```pycon
>>> from coola.summary import summarize
>>> print(summarize({"name": "Alice", "age": 30}))
<class 'dict'> (length=2)
  (name): Alice
  (age): 30

```

### Dataclasses and Pydantic Models

Dataclasses and pydantic models list their fields. The values are summarized like the items of a
dictionary, so `max_depth` and the summarizers of the field types apply. The fields that are
meant to be hidden are not shown: the fields of a dataclass declared with `field(repr=False)`, and
the fields of a pydantic model that are annotated as a secret (`SecretStr`, `SecretBytes`):

```pycon
>>> from dataclasses import dataclass
>>> from coola.summary import summarize
>>> @dataclass
... class Point:
...     x: float
...     y: float
>>> print(summarize(Point(1.0, 2.0)).replace(f"{Point}", "Point"))
Point (fields=2)
  (x): 1.0
  (y): 2.0

```

### Iterators

A generator or an iterator is a leaf: its `str` is shown and no item is taken from it, so the
summary never consumes the data.

## Working with Nested Structures

### Depth Control

The most important option of `coola.summary` is `max_depth`. It is a **display depth**: the number
of levels that are expanded. The structures deeper than `max_depth` are shown on one line (the
compact form), and nothing is raised. This is the opposite of the `max_depth` of
`objects_are_equal`, which is a **limit that raises** an error for a deeper structure:

```pycon
>>> from coola.summary import summarize
>>> data = {"level1": {"level2": {"level3": [1, 2, 3]}}}

```

**max_depth=0**: Shows the data on one line (no expansion). The nesting of the line is cut after
three levels:

```pycon
>>> print(summarize(data, max_depth=0))
{'level1': {'level2': {'level3': [...]}}}

```

**max_depth=1** (the default): Expands only the top level:

```pycon
>>> print(summarize(data, max_depth=1))
<class 'dict'> (length=1)
  (level1): {'level2': {'level3': [1, 2, 3]}}

```

**max_depth=2**: Expands two levels deep:

```pycon
>>> print(summarize(data, max_depth=2))
<class 'dict'> (length=1)
  (level1): <class 'dict'> (length=1)
    (level2): {'level3': [1, 2, 3]}

```

**max_depth=3**: Expands three levels deep:

```pycon
>>> print(summarize(data, max_depth=3))
<class 'dict'> (length=1)
  (level1): <class 'dict'> (length=1)
    (level2): <class 'dict'> (length=1)
      (level3): [1, 2, 3]

```

Each level is indented by two spaces. A text with several lines is aligned under its first
character. The summary of a structure that is nested 100,000 levels is built without a
`RecursionError`: the size of the text is bounded by `max_depth`, `max_items` and `max_characters`.

### The Compact Form

A node at the display depth is shown on one line, with a bounded work: a `list`, a `tuple`, a
`deque`, a `dict`, a `set` or a `frozenset` shows at most `max_items` items followed by `...`, the
nested containers are cut after three levels, and a long string is shortened in the middle. The
sub-structure is never converted to a string, so a huge list at the display depth costs as much as
a small one:

```pycon
>>> from coola.summary import summarize
>>> print(summarize({"values": list(range(1_000_000)), "nested": [[[[1]]]]}))
<class 'dict'> (length=2)
  (values): [0, 1, 2, 3, 4, ...]
  (nested): [[[[...]]]]

```

Any other type (a subclass of `list`, a `Counter`, ...) is shown with its `repr`, cut at
`max_characters`.

### Cycles

A container that contains itself is shown as `<cycle>` where it comes back. A container that is
reachable by several paths (it is not a cycle) is shown once per path:

```pycon
>>> from coola.summary import summarize
>>> data = [1]
>>> data.append(data)
>>> print(summarize(data, max_depth=3))
<class 'list'> (length=2)
  (0): <class 'int'> 1
  (1): <cycle>

```

### Errors in the Data

Summarizing never fails because of the data. A `__len__`, a property or a `__iter__` that raises is
replaced by a placeholder that names the exception type (never its message, which may carry data),
and the rest of the summary is intact:

```pycon
>>> from coola.summary import summarize
>>> class Broken(list):
...     def __len__(self):
...         raise RuntimeError("broken")
>>> print(summarize([Broken(), 1]).replace(f"{Broken}", "Broken"))
<class 'list'> (length=2)
  (0): Broken <summary failed: RuntimeError>
  (1): 1

```

A `__str__` or a `__repr__` that raises is replaced by `<ClassName object at 0x...>`.

## Controlling Output Size

### Limiting Items

By default, a node shows at most 5 children. The other ones are announced, never silently dropped:

```pycon
>>> from coola.summary import summarize
>>> print(summarize(list(range(20))))
<class 'list'> (length=20)
  (0): 0
  (1): 1
  (2): 2
  (3): 3
  (4): 4
  ... (+15 more)

```

Use `max_items` to change the limit. `max_items=None` shows all the items:

```pycon
>>> print(summarize(list(range(10)), max_items=3))
<class 'list'> (length=10)
  (0): 0
  (1): 1
  (2): 2
  ... (+7 more)

```

`max_items` applies to the items of the sequences, the mappings and the sets, to the fields of the
dataclasses and the pydantic models, and to the columns of the data frames, the tables and the
datasets.

### Truncating Long Texts

A text is cut at `max_characters` characters (1000 by default). The `str` of the object is computed
before it is cut. Use `max_characters=None` to show everything:

```pycon
>>> from coola.summary import summarize
>>> print(summarize("This is a very long string that is cut", max_characters=24))
<class 'str'> This is a...

```

## Working with NumPy and PyTorch

The default registry summarizes the arrays and the tensors with their metadata, so a large array
never floods the logs:

```pycon
>>> import numpy as np
>>> from coola.summary import summarize
>>> print(summarize(np.arange(100).reshape(10, 10)))
<class 'numpy.ndarray'> | shape=(10, 10) | dtype=int64
>>> import torch
>>> print(summarize(torch.randn(3, 4, 5)))
<class 'torch.Tensor'> | shape=torch.Size([3, 4, 5]) | dtype=torch.float32 | device=cpu | requires_grad=False

```

Use `show_data=True` to see a bounded `repr` of the values instead:

```pycon
>>> print(summarize(np.arange(5), show_data=True))
array([0, 1, 2, 3, 4])
>>> print(summarize(torch.arange(5), show_data=True))
tensor([0, 1, 2, 3, 4])

```

### DataFrames, Arrays and Other Array-Like Types

The default registry also summarizes the other array-like types supported by `coola.equality`,
when their package is installed. They show metadata only (type, shape, dtype and column names).
At most `max_items` columns or variables are listed, and the other ones are replaced by
`... (+N more)`. A lazy frame is never collected.

| Type                                    | Summarizer                  | Summary                                    |
| --------------------------------------- | --------------------------- | ------------------------------------------ |
| `pandas.DataFrame`                      | `PandasDataFrameSummarizer` | `shape`, `columns` with their dtype        |
| `pandas.Series`                         | `PandasSeriesSummarizer`    | `shape`, `dtype`, `name`                   |
| `polars.DataFrame`                      | `PolarsDataFrameSummarizer` | `shape`, `columns` with their dtype        |
| `polars.LazyFrame`                      | `PolarsLazyFrameSummarizer` | `columns` with their dtype (not collected) |
| `polars.Series`                         | `PolarsSeriesSummarizer`    | `shape`, `dtype`, `name`                   |
| `pyarrow.Table`, `pyarrow.RecordBatch`  | `PyarrowTableSummarizer`    | `shape`, `columns` with their type         |
| `pyarrow.Array`, `pyarrow.ChunkedArray` | `PyarrowArraySummarizer`    | `length`, `type`                           |
| `xarray.DataArray`                      | `XarrayDataArraySummarizer` | `dims`, `shape`, `dtype`, `name`           |
| `xarray.Dataset`                        | `XarrayDatasetSummarizer`   | `dims`, `data_vars` with their dtype       |
| `xarray.Variable`                       | `XarrayVariableSummarizer`  | `dims`, `shape`, `dtype`                   |
| `jax.Array`                             | `JaxArraySummarizer`        | `shape`, `dtype`                           |

```pycon
>>> import pandas as pd
>>> from coola.summary import summarize
>>> frame = pd.DataFrame({"a": [1], "b": [2.0], "c": [3]})
>>> print(summarize(frame))
<class 'pandas.DataFrame'> | shape=(1, 3) | columns=a: int64, b: float64, c: int64
>>> print(summarize(frame, max_items=2))
<class 'pandas.DataFrame'> | shape=(1, 3) | columns=a: int64, b: float64, ... (+1 more)

```

To add a summarizer for another type that is described by its metadata, extend
`BaseMetadataSummarizer` and implement `_describe`.

## Creating Custom Summarizers

### The Contract

A summarizer **describes one type**. It returns:

- a `str` for a leaf, or
- a `Node(header, children, total)` for a container: the first line, the lazy `(label, child)`
  pairs in display order, and the number of children.

The engine owns the recursion, the depth, the limits, the cycles and the errors, so a summarizer
does not call the registry and cannot forget a protection. The rules are:

1. `summarize(data, config, *, compact)` is pure and cheap: it does not recurse, does not modify the
   data and does not iterate the whole data. It reads `len(data)` and returns the children lazily
   (`enumerate(data)`, `data.items()`, a generator): the engine takes at most `max_items` of them.
2. `compact` is `True` when the data is at the display depth. A leaf may use it. A container
   ignores it: the engine shows it on one line.
3. A leaf may return a text with several lines. The engine cuts it at `max_characters`.
4. The labels of the children can be any object: the engine writes their `str`.
5. An exception is handled by the engine, which writes a placeholder.

`config` is a `SummaryConfig`, the frozen and validated object that holds `max_depth`, `max_items`,
`max_characters` and `show_data`. A protocol (`Summarizer`) is enough: `BaseSummarizer` is a
convenience base class.

### A Leaf

```pycon
>>> from coola.summary import BaseSummarizer, register_summarizers, summarize
>>> class Person:
...     def __init__(self, name, age):
...         self.name = name
...         self.age = age
>>> class PersonSummarizer(BaseSummarizer[Person]):
...     def summarize(self, data, config, *, compact):
...         return f"Person(name={data.name!r}, age={data.age})"
>>> register_summarizers({Person: PersonSummarizer()})
>>> print(summarize(Person("Alice", 30)))
Person(name='Alice', age=30)

```

### A Container

```pycon
>>> from coola.summary import BaseSummarizer, Node, register_summarizers, summarize
>>> class Tree:
...     def __init__(self, *children):
...         self.children = children
>>> class TreeSummarizer(BaseSummarizer[Tree]):
...     def summarize(self, data, config, *, compact):
...         n = len(data.children)
...         return Node(f"Tree (children={n})", enumerate(data.children), n)
>>> register_summarizers({Tree: TreeSummarizer()})
>>> print(summarize(Tree(1, Tree(2, 3)), max_depth=2))
Tree (children=2)
  (0): <class 'int'> 1
  (1): Tree (children=2)
    (0): 2
    (1): 3

```

The summarizers can also be provided by a package, with the entry-point group
`coola.summary.summarizers`.

## Working with the Registry

### Understanding the Registry

The `SummarizerRegistry` manages the mapping from types to summarizers. `find_summarizer` returns the
summarizer of a type, or of its nearest registered parent type along the Method Resolution Order.
The full content of the global registry depends on the installed libraries and on the summarizers
registered so far, so it is not shown here:

```pycon
>>> from coola.summary import get_default_registry
>>> registry = get_default_registry()
>>> registry.find_summarizer(list)
SequenceSummarizer()
>>> registry.find_summarizer(dict)
MappingSummarizer()
>>> registry.find_summarizer(str)
DefaultSummarizer()

```

### Creating a Custom Registry

For complete control, create your own registry and pass it to `summarize()`. The registry must
contain a summarizer for `object`, which is the catch-all:

```pycon
>>> from coola.summary import SummarizerRegistry, SequenceSummarizer, DefaultSummarizer, summarize
>>> registry = SummarizerRegistry({object: DefaultSummarizer(), list: SequenceSummarizer()})
>>> print(summarize([1, 2, 3], registry=registry))
<class 'list'> (length=3)
  (0): 1
  (1): 2
  (2): 3

```

### Registering Multiple Types

Use `register_many()` to register multiple types at once:

```pycon
>>> from coola.summary import (
...     SummarizerRegistry,
...     SequenceSummarizer,
...     MappingSummarizer,
...     DefaultSummarizer,
... )
>>> registry = SummarizerRegistry()
>>> registry.register_many(
...     {object: DefaultSummarizer(), list: SequenceSummarizer(), dict: MappingSummarizer()}
... )

```

### Checking Registered Types

`has_summarizer` checks if a summarizer is registered for exactly this type. `find_summarizer`
resolves the type with its parents, so `int` is not registered but has the summarizer of `object`:

```pycon
>>> from coola.summary import get_default_registry
>>> registry = get_default_registry()
>>> registry.has_summarizer(list)
True
>>> registry.has_summarizer(int)
False

```

## Common Use Cases

### Debugging Complex Data

Quickly inspect complex data structures during debugging:

```pycon
>>> from coola.summary import summarize
>>> config = {
...     "database": {"host": "localhost", "port": 5432},
...     "cache": {"enabled": True, "ttl": 3600, "backends": ["redis", "memcached"]},
... }
>>> print(summarize(config))
<class 'dict'> (length=2)
  (database): {'host': 'localhost', 'port': 5432}
  (cache): {'enabled': True, 'ttl': 3600, 'backends': ['redis', 'memcached']}

```

### Logging Large Tensors

Log tensor shapes without filling logs with data:

```pycon
>>> import torch
>>> from coola.summary import summarize
>>> model_weights = {"layer1": torch.randn(1000, 1000), "bias": torch.randn(500)}
>>> print(summarize(model_weights))
<class 'dict'> (length=2)
  (layer1): <class 'torch.Tensor'> | shape=torch.Size([1000, 1000]) | dtype=torch.float32 | device=cpu | requires_grad=False
  (bias): <class 'torch.Tensor'> | shape=torch.Size([500]) | dtype=torch.float32 | device=cpu | requires_grad=False

```

### Comparing Data Structures

Get a quick overview to compare different data structures. With the default `max_depth=1`, the
nested structures are shown on one line, and the collections that have more than `max_items` items
are cut:

```pycon
>>> from coola.summary import summarize
>>> data = {"users": list(range(20)), "version": "1.0"}
>>> print(summarize(data))
<class 'dict'> (length=2)
  (users): [0, 1, 2, 3, 4, ...]
  (version): 1.0

```

For a deeper inspection, increase `max_depth`:

```pycon
>>> print(summarize(data, max_depth=2, max_items=3))
<class 'dict'> (length=2)
  (users): <class 'list'> (length=20)
    (0): 0
    (1): 1
    (2): 2
    ... (+17 more)
  (version): <class 'str'> 1.0

```

## Available Summarizers

The `coola.summary` package provides the following built-in summarizers. None of them has an
option: the options are the arguments of `summarize()`.

- **`DefaultSummarizer`**: For generic objects and scalar types (registered for `object` and
  `UserString`)
- **`SequenceSummarizer`**: For sequences (`list`, `tuple`, `deque`, the `Sequence` ABC)
- **`MappingSummarizer`**: For mappings (`dict`, the `Mapping` ABC, `MappingProxyType`)
- **`SetSummarizer`**: For sets (`set`, `frozenset`)
- **`DataclassSummarizer`**: For dataclasses
- **`PydanticModelSummarizer`**: For pydantic models (requires pydantic)
- **`NDArraySummarizer`**, **`TensorSummarizer`**, **`PandasDataFrameSummarizer`**,
  **`PandasSeriesSummarizer`**, **`PolarsDataFrameSummarizer`**, **`PolarsLazyFrameSummarizer`**,
  **`PolarsSeriesSummarizer`**, **`PyarrowArraySummarizer`**, **`PyarrowTableSummarizer`**,
  **`XarrayDataArraySummarizer`**, **`XarrayDatasetSummarizer`**, **`XarrayVariableSummarizer`**,
  **`JaxArraySummarizer`**: For the array-like types (require the corresponding package). They
  extend `BaseMetadataSummarizer`.

`format_fields()` formats `(name, type)` pairs as `name: type`, limited to a maximum number of
pairs. `get_default_summarizer_registry()` returns the registry used by `summarize()`.

## Design Principles

The `coola.summary` package is designed with the following principles:

1. **One engine, one extension point, one option object**: the engine owns the invariants, a
   summarizer describes one type, and `SummaryConfig` holds the options
2. **Type-based dispatch**: Automatically selects the appropriate summarizer based on data type
3. **Configurable output**: Control depth, item limits, and text length to suit your needs
4. **Extensibility**: Easy to add support for custom types via the registry
5. **Sensible defaults**: Works out-of-the-box for common Python types
6. **Metadata focus**: For large data structures (arrays, tensors), show metadata instead of data

## See Also

- [Recursive Data Transformation](recursive.md): For transforming nested data while preserving structure
- [Iterating Over Nested Data](iterator.md): For iterating over nested data structures
- [Registry System](registry.md): For understanding the registry pattern used internally

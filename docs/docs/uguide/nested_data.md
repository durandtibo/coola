# Nested Data and Leaves

:book: This page explains what kind of data `coola` works with: **nested data**, made of
containers and leaves. Read it first to understand what the other packages (`equality`, `hashing`,
`summary`, `recursive`, `iterator`, `nested`) accept and how they treat each value.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## What is nested data?

Nested data is a value that contains other values, which may contain other values, and so on. A
configuration, the output of a model, a batch of tensors or a JSON document are nested data:

```python
batch = {
    "id": 42,
    "name": "run-1",
    "inputs": [np.zeros((2, 3)), np.ones((2, 3))],
    "metrics": {"loss": 0.25, "tags": ("a", "b")},
}
```

`coola` was built to compare, hash, summarize, transform and traverse such objects as a whole,
whatever the libraries used inside (Python, NumPy, PyTorch, pandas, ...).

## Containers and leaves

Every value of a nested structure is one of two things:

- A **container** holds other values, called its **children**: a `list`, a `tuple`, a `dict`, a
  `set`, ... `coola` goes inside a container and looks at its children.
- A **leaf** is a value that `coola` does not open: a number, a string, a NumPy array, a
  DataFrame, ... The leaf is handled as a whole, by the code that knows its type (for example,
  the NumPy tester compares two arrays with `allclose`).

```pycon
>>> from coola.iterator import dfs_iterate
>>> list(dfs_iterate({"a": 1, "b": [2, 3], "c": {"d": "text"}}))
[1, 2, 3, 'text']

```

The dictionary and the lists are containers; `1`, `2`, `3` and `'text'` are the leaves. Only the
leaves are returned: the containers are never returned, and an empty container has no leaf.

Two rules apply to every package of `coola`:

- **The type decides.** Whether a value is a container or a leaf depends on its type (and its
  parent classes), not on its content. A `list` is always a container, even if it is empty; an
  `int` is always a leaf.
- **When in doubt, it is a leaf.** A type that `coola` does not know is a leaf. An object is never
  opened by guessing.

## The containers

These types are containers in every package:

| Type                                                                 | Children                                                   |
| -------------------------------------------------------------------- | ---------------------------------------------------------- |
| `list`, `tuple` (and named tuples), `collections.deque`              | The elements, in order                                     |
| Other sequences (`collections.abc.Sequence`, `collections.UserList`) | The elements, in order                                     |
| `dict`, `collections.OrderedDict`, `defaultdict`, `Counter`          | The values, in the order of the keys                       |
| Other mappings (`collections.abc.Mapping`, `MappingProxyType`)       | The values                                                 |
| `set`, `frozenset`                                                   | The elements (their order is not specified)                |
| Dataclasses                                                          | The values of the fields, in the order of their definition |

The **keys** of a mapping are not children: they are labels. They are compared by `equality`, but
`iterator` and `recursive` never return or transform them.

Containers can be mixed and nested at any depth:

```pycon
>>> from collections import deque
>>> from dataclasses import dataclass
>>> from coola.iterator import dfs_iterate
>>> @dataclass
... class Config:
...     lr: float
...     layers: list
>>> data = {"config": Config(0.1, [64, 32]), "queue": deque([1, 2]), "seen": (3, (4,))}
>>> list(dfs_iterate(data))
[0.1, 64, 32, 1, 2, 3, 4]

```

## The leaves

Everything that is not a container is a leaf. The most common leaves are:

| Kind                  | Examples                                                                           |
| --------------------- | ---------------------------------------------------------------------------------- |
| Scalars               | `int`, `float`, `complex`, `bool`, `None`, `decimal.Decimal`, `fractions.Fraction` |
| Text and bytes        | `str`, `bytes` (a string is not a container of characters)                         |
| Dates and identifiers | `datetime.datetime`, `datetime.date`, `pathlib.Path`, `uuid.UUID`, `enum.Enum`     |
| Arrays and tensors    | `numpy.ndarray`, `torch.Tensor`, `jax.Array`, `xarray.DataArray`, `array.array`    |
| Tables                | `pandas.DataFrame` and `Series`, `polars.DataFrame` and `Series`, `pyarrow.Table`  |
| Iterators             | Generators and iterators: they are returned as they are and never consumed         |
| Other objects         | Any instance of a class that `coola` does not know                                 |

A NumPy array or a DataFrame holds many values, but `coola` does not treat it as a container of
numbers: it is one leaf, compared, hashed and summarized by a dedicated handler. This is the point
of the package: the structure around the arrays is handled by `coola`, and the arrays are handled
by the library that knows them.

```pycon
>>> import numpy as np
>>> from coola.iterator import dfs_iterate
>>> leaves = list(dfs_iterate({"x": np.zeros(3), "y": [np.ones(2), "text"]}))
>>> [type(leaf).__name__ for leaf in leaves]
['ndarray', 'ndarray', 'str']

```

Optional libraries (NumPy, PyTorch, pandas, polars, PyArrow, JAX, xarray) are **never required**:
`coola` uses a library only if it is installed, and importing `coola` does not import them.

## What each package does with nested data

The packages share the same idea of container and leaf, and each adds what it does with the leaves:

| Package                     | Does what with the leaves                                                              |
| --------------------------- | -------------------------------------------------------------------------------------- |
| [`equality`](equality.md)   | Compares two nested objects leaf by leaf (`objects_are_equal`, `objects_are_allclose`) |
| [`hashing`](hashing.md)     | Computes a stable hash of the whole structure                                          |
| [`summary`](summary.md)     | Builds a short, readable description, with a depth limit                               |
| [`recursive`](recursive.md) | Applies a function to each leaf and rebuilds the same structure                        |
| [`iterator`](iterator.md)   | Returns the leaves one at a time, depth first or breadth first                         |
| [`nested`](nested.md)       | Utilities built on these (flatten, filter and rename keys, merge, ...)                 |
| [`jsonable`](jsonable.md)   | Converts the structure to JSON data (`to_jsonable`)                                    |

```pycon
>>> from coola.equality import objects_are_equal
>>> from coola.recursive import recursive_apply
>>> data = {"a": 1, "b": [2, 3]}
>>> recursive_apply(data, lambda x: x * 10)
{'a': 10, 'b': [20, 30]}
>>> objects_are_equal(data, {"a": 1, "b": [2, 3]})
True

```

## Properties shared by all packages

- **Any depth.** A structure can be nested as deeply as memory allows; `recursive` and `iterator`
  do not hit the Python recursion limit.
- **Cycles are detected.** A container that contains itself raises a `ValueError` instead of
  looping forever (`equality` treats the pair as equal).
- **A shared container is visited once per path.** `[shared, shared]` gives the leaves of `shared`
  twice.
- **The input is never modified.**
- **The same data gives the same leaves** in `equality`, `hashing`, `summary`, `recursive` and
  `iterator`.

## Supporting your own types

If a type is not known, it is a leaf. There are two ways to change that:

- To **handle** a leaf type with a specific behaviour (compare two instances of a class, hash
  them, ...), register a handler for the type in the registry of the package. See
  [Extending coola](extending.md) and the [registry](registry.md) guide.
- To **open** a custom container, register a handler that returns its children, for example a
  child finder for `iterator` or a transformer for `recursive`.

A handler registered for a type also applies to its subclasses, and the most specific type wins.

## See Also

- [`coola.equality`](equality.md): Compare nested data
- [`coola.iterator`](iterator.md): Iterate over the leaves
- [`coola.recursive`](recursive.md): Transform the leaves
- [`coola.registry`](registry.md): The type dispatch behind all of them

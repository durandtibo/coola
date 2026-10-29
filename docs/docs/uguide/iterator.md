# Iterating Over Nested Data

:book: This page describes the `coola.iterator` package, which provides utilities for iterating
over nested data structures using different traversal strategies.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.iterator` package provides functions to iterate over nested data structures (lists,
dicts, tuples, sets, etc.) using two different traversal strategies:

- **Depth-First Search (DFS)**: Traverses nested structures by going deep into each branch before
  moving to the next
- **Breadth-First Search (BFS)**: Traverses nested structures level by level

The package also provides filtering utilities to extract only specific types from heterogeneous
collections.

## Properties

| Property                     | Statement                                                                                                                                                                                                                                     |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Leaves only                  | A traversal yields the leaves, once per occurrence, never a container. An empty container has no leaf and yields nothing.                                                                                                                     |
| Type dispatch                | The iterator (or child finder) is chosen from the type of the data along the Method Resolution Order, so the most specific registered type wins.                                                                                              |
| Deterministic order          | DFS is pre-order and BFS is level order, both left to right. A mapping gives its values in insertion order (the keys are never yielded) and a dataclass its fields in the order of their definition. The order inside a set is not specified. |
| Same leaves, different order | DFS and BFS visit the same leaves. Only the order differs.                                                                                                                                                                                    |
| Lazy                         | Both `dfs_iterate` and `bfs_iterate` are generators: leaves are produced one at a time, and nothing is walked before the first `next`.                                                                                                        |
| Read-only                    | The input is never modified.                                                                                                                                                                                                                  |
| Every leaf is returned       | A container that appears several times, without containing itself, gives its leaves each time.                                                                                                                                                |
| Cycle-safe                   | A container that contains itself raises `ValueError` instead of looping forever or hitting the recursion limit.                                                                                                                               |
| Iterators are leaves         | The default registries never consume an iterator or a generator, so an infinite iterator cannot hang the traversal.                                                                                                                           |
| Depth                        | Both functions use an explicit stack or queue: there is no `RecursionError`, whatever the nesting. The depth is bounded by memory only.                                                                                                       |
| Extensible                   | Support for a new container type is added with one child finder (`find_children`) registered for the type. It serves both orders.                                                                                                             |
| Errors propagate             | An exception of a child finder, or of the data (for example a failing property), propagates unchanged.                                                                                                                                        |
| Thread-safe                  | The state of a traversal (stack or queue, cache, ancestors) is local to the call, so a registry and a child finder can be shared by threads. A running traversal must not be shared.                                                          |
| Consistent with siblings     | The containers entered are the ones of `recursive_apply`; dataclasses are entered as in `objects_are_equal`, `hash_object` and `summarize`; the options are keyword-only; the DFS order is the order of the calls of `recursive_apply`.       |

Not guaranteed:

- **Shared references.** A container reachable by several paths (a DAG) gives its leaves once per
  path, so the cost is proportional to the number of paths, not to the number of objects.
- **The order inside a set**, and the order of a `ChainMap` (its values follow the lookup order of
  the maps, last map first).
- **Mutation during the traversal.** Changing a container while it is walked is the caller's
  problem, as for any Python iteration.
- **A registry modified during a traversal.** The traversal keeps the child finders it already
  resolved.

What the package does not do: pydantic models, attrs classes and plain objects with `__dict__` are
leaves, and so are NumPy arrays, tensors and data frames (the function receives the whole array;
element-wise iteration is the job of the caller). There is no `max_depth` or `max_leaves` option:
`itertools.islice` stops a traversal after `n` leaves, because the functions are lazy. The cycle is
reported when the traversal reaches it, so the leaves before it have already been yielded.

## Depth-First Search (DFS) Iteration

### Basic Usage

The `dfs_iterate` function performs depth-first traversal of nested data structures:

```pycon
>>> from coola.iterator import dfs_iterate
>>> list(dfs_iterate({"a": 1, "b": "abc"}))
[1, 'abc']
>>> list(dfs_iterate([1, [2, 3], {"x": 4}]))
[1, 2, 3, 4]

```

The function yields only the leaf values (atomic values that are not containers), excluding the
containers themselves even if they're empty.

### DFS Traversal Examples

**Simple nested lists:**

```pycon
>>> from coola.iterator import dfs_iterate
>>> list(dfs_iterate([1, 2, [3, 4, [5, 6]]]))
[1, 2, 3, 4, 5, 6]

```

**Nested dictionaries:**

```pycon
>>> from coola.iterator import dfs_iterate
>>> data = {"level1": {"level2": {"level3": 42}}}
>>> list(dfs_iterate(data))
[42]

```

**Mixed structures:**

```pycon
>>> from coola.iterator import dfs_iterate
>>> data = {
...     "list": [1, 2, 3],
...     "dict": {"a": 4, "b": 5},
...     "set": {6, 7},
...     "value": 8,
... }
>>> sorted(dfs_iterate(data))  # sorted for consistent output
[1, 2, 3, 4, 5, 6, 7, 8]

```

**Sets and tuples:**

```pycon
>>> from coola.iterator import dfs_iterate
>>> list(dfs_iterate((1, (2, 3))))
[1, 2, 3]
>>> list(dfs_iterate((1, (2, (3, 4)))))
[1, 2, 3, 4]

```

### Understanding DFS Order

Depth-first search goes deep into each branch before moving to the next branch:

```pycon
>>> from coola.iterator import dfs_iterate
>>> data = {
...     "branch1": [1, [2, 3]],
...     "branch2": [4, [5, 6]],
... }
>>> # DFS will fully explore branch1 before moving to branch2
>>> list(dfs_iterate(data))
[1, 2, 3, 4, 5, 6]

```

## Breadth-First Search (BFS) Iteration

### Basic Usage

The `bfs_iterate` function performs breadth-first traversal of nested data structures:

```pycon
>>> from coola.iterator import bfs_iterate
>>> list(bfs_iterate({"a": 1, "b": "abc"}))
[1, 'abc']
>>> list(bfs_iterate([1, [2, 3], {"x": 4}]))
[1, 2, 3, 4]

```

Like `dfs_iterate`, it yields only the leaf values, excluding containers.

### BFS Traversal Examples

**Nested lists:**

```pycon
>>> from coola.iterator import bfs_iterate
>>> list(bfs_iterate([1, [2, [3, [4]]]]))
[1, 2, 3, 4]

```

**Multi-level structure:**

```pycon
>>> from coola.iterator import bfs_iterate
>>> data = {
...     "level1_a": 1,
...     "level1_b": {"level2_a": 2, "level2_b": {"level3": 3}},
... }
>>> # BFS processes level by level: 1 at level 1, then 2 at level 2, then 3 at level 3
>>> list(bfs_iterate(data))
[1, 2, 3]

```

### BFS vs DFS

The main difference between BFS and DFS is the order of traversal:

```pycon
>>> from coola.iterator import bfs_iterate, dfs_iterate
>>> data = [[1, 2], [3, 4]]
>>> list(dfs_iterate(data))  # DFS: depth-first
[1, 2, 3, 4]
>>> list(bfs_iterate(data))  # BFS: breadth-first
[1, 2, 3, 4]

```

For this simple example, the results are the same, but for more complex nested structures, the
traversal order can differ significantly. BFS processes all items at each level before moving
deeper, while DFS goes as deep as possible before backtracking.

### Dataclasses

`dfs_iterate` and `bfs_iterate` enter the dataclass instances, like `objects_are_equal`,
`hash_object`, `summarize` and `recursive_apply`: they visit the values of the fields, in the order
of their definition (`DataclassChildFinder`):

```pycon
>>> from dataclasses import dataclass
>>> from coola.iterator import dfs_iterate
>>> @dataclass
... class Config:
...     lr: float
...     layers: list
>>> list(dfs_iterate(Config(0.1, [1, 2])))
[0.1, 1, 2]

```

A child finder registered for the type of the dataclass has priority over the default one. To keep
a dataclass as a leaf, register `DefaultChildFinder()` for its type:

```pycon
>>> from coola.iterator import DefaultChildFinder, get_default_registry
>>> registry = get_default_registry().copy()
>>> registry.register(Config, DefaultChildFinder())
>>> list(dfs_iterate(Config(0.1, [1, 2]), registry=registry))
[Config(lr=0.1, layers=[1, 2])]

```

## Paths of the Leaves

The `dfs_items` function is `dfs_iterate` with the path of each leaf: it yields the pairs
`(path, leaf)` in the same order, where `path` is the tuple of the labels of the containers crossed to
reach the leaf. The label of a child is the key of a mapping (not converted), the index of a
sequence, the name of the field of a dataclass, or the element itself for a set (a set has no
position). The path of a root that is a leaf is the empty tuple.

```pycon
>>> from coola.iterator import dfs_items
>>> list(dfs_items({"a": 1, "b": [2, {"c": 3}]}))
[(('a',), 1), (('b', 0), 2), (('b', 1, 'c'), 3)]

```

By default an empty container yields nothing, like in `dfs_iterate`. The keyword-only option
`empty=True` also yields an empty container as a leaf, so it is not lost:

```pycon
>>> from coola.iterator import dfs_items
>>> list(dfs_items({"a": {}, "b": 1}))
[(('b',), 1)]
>>> list(dfs_items({"a": {}, "b": 1}, empty=True))
[(('a',), {}), (('b',), 1)]

```

The labels come from the `find_items` method of the child finder. The base class derives it from
`find_children` by numbering the children, so a custom child finder works with `dfs_items` without
change; override `find_items` to give the real key or name of each child.

## Type-Based Filtering

### Basic Filtering

The `filter_by_type` function filters an iterator to yield only values of specified types:

```pycon
>>> from coola.iterator import filter_by_type
>>> list(filter_by_type([1, "hello", 2, 3.14, "world", 4], int))
[1, 2, 4]

```

### Filtering Multiple Types

You can filter for multiple types using a tuple (the type checker infers the union of the types):

```pycon
>>> from coola.iterator import filter_by_type
>>> # Note: bool is a subclass of int
>>> list(filter_by_type([1, "hello", 2.5, True, None, [1, 2]], (int, float)))
[1, 2.5, True]

```

### Combining with Iteration

You can combine type filtering with nested iteration:

```pycon
>>> from coola.iterator import dfs_iterate, filter_by_type
>>> data = {"a": 1, "b": "hello", "c": [2, "world", 3.14]}
>>> # Get all numeric values (int and float) from nested structure
>>> list(filter_by_type(dfs_iterate(data), (int, float)))
[1, 2, 3.14]

```

## Advanced Usage

### Writing a Child Finder

A child finder inherits from `BaseChildFinder` and implements `find_children(data)`, which returns
the direct children of the object in the order they are visited, or `None` for a leaf. It does not
call the registry and does not recurse: `dfs_iterate` and `bfs_iterate` walk the children, so one
child finder serves both orders. An empty container returns an empty iterable (it yields nothing),
a leaf returns `None` (the value itself is yielded). `MappingChildFinder` returns the values of a
mapping, and `IterableChildFinder` the container itself:

```pycon
>>> from coola.iterator import (
...     BaseChildFinder,
...     MappingChildFinder,
...     bfs_iterate,
...     dfs_iterate,
...     register_child_finders,
... )
>>> list(MappingChildFinder().find_children({"a": 1, "b": 2}))
[1, 2]
>>> class Node:
...     def __init__(self, left, right):
...         self.left, self.right = left, right
>>> class NodeChildFinder(BaseChildFinder[Node]):
...     def find_children(self, data):
...         return [data.left, data.right]
>>> register_child_finders({Node: NodeChildFinder()})
>>> tree = Node(Node(1, 2), 3)
>>> list(dfs_iterate(tree))
[1, 2, 3]
>>> list(bfs_iterate(tree))
[3, 1, 2]

```

### Default Registry

`get_default_registry()` (alias `get_default_child_finder_registry()`) returns the process-wide
registry used by `dfs_iterate` and `bfs_iterate`. `register_child_finders()` registers types on
this registry. The default registry enters `list`, `tuple`, `deque`, `Sequence`, `set`,
`frozenset`, `dict`, `Mapping`, `MappingProxyType` and the dataclasses (the containers of
`recursive_apply`); everything else is a leaf.

### Using Custom Registry

You can create and use a custom registry:

```pycon
>>> from coola.iterator import (
...     ChildFinderRegistry,
...     DefaultChildFinder,
...     IterableChildFinder,
...     dfs_iterate,
... )
>>> registry = ChildFinderRegistry()
>>> registry.register(list, IterableChildFinder())
>>> registry.register(object, DefaultChildFinder())
>>> list(dfs_iterate([1, 2, 3], registry=registry))
[1, 2, 3]

```

## Common Use Cases

### Extracting All Values

Extract all values from a complex configuration:

```pycon
>>> from coola.iterator import dfs_iterate
>>> config = {
...     "database": {"host": "localhost", "port": 5432},
...     "cache": {"enabled": True, "ttl": 3600},
... }
>>> list(dfs_iterate(config))
['localhost', 5432, True, 3600]

```

### Counting Values

Count specific types in a nested structure:

```pycon
>>> from coola.iterator import dfs_iterate, filter_by_type
>>> data = {
...     "scores": [95, 87, 92],
...     "names": ["Alice", "Bob", "Charlie"],
... }
>>> len(list(filter_by_type(dfs_iterate(data), int)))
3

```

### Finding All Strings

Find all string values in a nested structure:

```pycon
>>> from coola.iterator import dfs_iterate, filter_by_type
>>> data = {
...     "user": {"name": "John", "age": 30},
...     "posts": [{"title": "Post 1", "views": 100}, {"title": "Post 2", "views": 200}],
... }
>>> list(filter_by_type(dfs_iterate(data), str))
['John', 'Post 1', 'Post 2']

```

### Validating Data Types

Check if all numeric values in a structure are within a range:

```pycon
>>> from coola.iterator import dfs_iterate, filter_by_type
>>> data = {"values": [10, 20, 30], "nested": {"more": [40, 50]}}
>>> numbers = list(filter_by_type(dfs_iterate(data), (int, float)))
>>> all(0 <= n <= 100 for n in numbers)
True

```

## Iterators and Cycles

The default registry expands the containers (`list`, `tuple`, `set`, `dict`, ...) but treat
iterators and generators as leaves: they are returned as they are and never consumed, so a
generator (or an infinite iterator) in the data cannot be exhausted or hang the traversal.

A container that contains itself cannot be traversed. Both `dfs_iterate` and `bfs_iterate` raise a
`ValueError` instead of looping forever or hitting the recursion limit. A container that appears
several times without containing itself is not a cycle, and its leaves are returned each time:

```pycon
>>> from coola.iterator import dfs_iterate
>>> shared = [1, 2]
>>> list(dfs_iterate([shared, shared]))
[1, 2, 1, 2]
>>> data = [1]
>>> data.append(data)
>>> list(dfs_iterate(data))
Traceback (most recent call last):
    ...
ValueError: Cycle detected: an object of type <class 'list'> contains itself

```

A child finder for a leaf type returns `None` (like `DefaultChildFinder`), so both functions yield
the value instead of expanding it.

## Containers and Leaves

The default registry gives the same containers as `recursive_apply`, so the same data gives the
same leaves in both packages:

| Type                                                                              | Children                                 |
| --------------------------------------------------------------------------------- | ---------------------------------------- |
| `list`, `tuple` (and named tuples), `deque`, `Sequence`, `set`, `frozenset`       | the elements, in iteration order         |
| `dict`, `Mapping`, `MappingProxyType`                                             | the values (the keys are never yielded)  |
| dataclass instances                                                               | the values of the fields, in field order |
| `str`, `bytes`, `UserString`, numbers, `None`, iterators, generators, arrays, ... | none: the value is a leaf                |

`range`, and a user class that only inherits `Collection` or `collections.abc.Set`, are leaves
until a child finder is registered for them (see [Writing a Child Finder](#writing-a-child-finder)).
An empty container has no leaf and yields nothing.

## Design Principles

The `coola.iterator` package design provides:

1. **Multiple traversal strategies**: Choose between DFS and BFS based on your needs
2. **Generator-based**: The leaves are produced on demand (a BFS keeps one level in memory)
3. **Type-aware**: Built-in support for common Python types with extensibility
4. **Clean API**: Simple functions that compose well with other Python tools

## See Also

- [`coola.recursive`](recursive.md): For transforming nested data while preserving structure
- [`coola.registry`](registry.md): For understanding the registry pattern used internally

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
>>> sorted(dfs_iterate({1, 2, {3, 4}}))  # sorted for consistent output
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
>>> list(dfs_iterate(data))  # Note: dict order may vary in Python < 3.7
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

## Type-Based Filtering

### Basic Filtering

The `filter_by_type` function filters an iterator to yield only values of specified types:

```pycon

>>> from coola.iterator import filter_by_type
>>> list(filter_by_type([1, "hello", 2, 3.14, "world", 4], int))
[1, 2, 4]

```

### Filtering Multiple Types

You can filter for multiple types using a tuple:

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

**Extract only strings:**

```pycon

>>> from coola.iterator import dfs_iterate, filter_by_type
>>> data = [1, "a", [2, "b", [3, "c"]]]
>>> list(filter_by_type(dfs_iterate(data), str))
['a', 'b', 'c']

```

**Extract only integers:**

```pycon

>>> from coola.iterator import bfs_iterate, filter_by_type
>>> data = {"nums": [1, 2.5, 3], "text": "hello", "flag": True}
>>> list(filter_by_type(bfs_iterate(data), int))
[1, 3, True]

```

## Advanced Usage

### Custom Iterators

For more control over how specific types are iterated, you can register custom iterators using the
registry system.

**DFS Custom Registry:**

```pycon

>>> from coola.iterator.dfs import register_iterators, IterableIterator
>>> # Register custom behavior for specific types
>>> register_iterators({list: IterableIterator()}, exist_ok=True)

```

**BFS Custom Registry:**

```pycon

>>> from coola.iterator.bfs import register_child_finders, IterableChildFinder
>>> # Register custom behavior for specific types
>>> register_child_finders({list: IterableChildFinder()}, exist_ok=True)

```

### Using Custom Registry

You can create and use a custom registry:

```pycon

>>> from coola.iterator.dfs import IteratorRegistry, dfs_iterate
>>> from coola.iterator.dfs import DefaultIterator, IterableIterator
>>> registry = IteratorRegistry()
>>> registry.register(list, IterableIterator())
>>> registry.register(object, DefaultIterator())
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
...     "passed": True,
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

## Design Principles

The `coola.iterator` package design provides:

1. **Multiple traversal strategies**: Choose between DFS and BFS based on your needs
2. **Generator-based**: Memory-efficient iteration without loading entire structures
3. **Type-aware**: Built-in support for common Python types with extensibility
4. **Clean API**: Simple functions that compose well with other Python tools

## See Also

- [`coola.recursive`](recursive.md): For transforming nested data while preserving structure
- [`coola.registry`](registry.md): For understanding the registry pattern used internally

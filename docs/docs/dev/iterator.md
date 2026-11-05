# Architecture of `coola.iterator`

:book: This page describes the design of the `coola.iterator` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/iterator.md#properties); the API is in the [reference](../refs/iterator.md).

## Constraints

1. **Simple and easy to maintain.** The package is small: functions that walk the same tree in
   different ways share one registry, one set of child finders, one default singleton and one plugin
   group, instead of one parallel package per order.
2. **No slower than the benchmark gate.** `dfs_iterate` and `bfs_iterate` are the primitives of the
   traversal helpers and of the documentation examples. A design that is simpler but slower is
   rejected.
3. **Consistent with the siblings.** The same input is handled the same way by `equality`,
   `hashing`, `summary`, `recursive` and `iterator`: the same containers are entered, dataclasses
   are entered, iterators are leaves, cycles are detected, options are keyword-only, and the order
   of `dfs_iterate` is the order in which `recursive_apply` calls `func`.
4. **Optional libraries stay lazy.** The package imports no optional library; NumPy arrays, pandas
   frames, ... are leaves unless a child finder is registered (the plugin group is the extension
   point).

## Architecture

```
dfs_iterate(data, *, registry=None)   bfs_iterate(data, *, registry=None)   dfs_items(data, *, registry=None, empty=False)
        |   one registry (default: the singleton), one call = one engine run
        v
engine.depth_first / breadth_first / depth_first_items(data, registry)
        |   coola/iterator/engine.py
        |   explicit stack / queue; per-call type -> finder cache; the containers on the path
        |
        |-- finder   = registry.find_child_finder(type(item))      MRO, fallbacks
        |-- children = finder.find_children(item)                  pure: no registry, no recursion
        |        None      -> leaf: yield item
        |        iterable  -> container: walk the children (DFS: push; BFS: enqueue)
        v
the leaves, one at a time
```

The engine owns the invariants (dispatch cache, leaf rule, cycles, depth, order, laziness). A child
finder **only describes the shape of a type**: its children. That is the same split as `recursive`
(`expand`) and `equality` (a tester returns the child pairs). The registry is not an argument of the
finder, so a finder cannot forget a protection and cannot recurse. Both orders call the same finder,
so "same leaves" is true by construction.

### The child finder contract

```python
class ChildFinder(Protocol[T_contra]):
    def find_children(self, data: T_contra) -> Iterable[Any] | None:
        """``None`` for a leaf; otherwise the children of ``data`` in the order they are visited."""


class BaseChildFinder(ABC, Generic[T]):  # convenience base class
    @abstractmethod
    def find_children(self, data: T) -> Iterable[Any] | None: ...

    def find_items(self, data: T) -> Iterable[tuple[Any, Any]] | None:
        """The labelled children ``(label, child)``; by default ``enumerate(find_children(data))``."""
```

Rules of the contract (documented in `BaseChildFinder`, checked by one parametrised unit test over
the default registry, like `NATIVE_SAMPLES` in `equality`):

1. `find_children` is pure and cheap: it does not call the registry, does not recurse, does not
   modify `data` and does not consume the children (it returns them).
2. The result is consumed once, lazily, in order. A finder can return the container itself
   (`IterableChildFinder` returns `data`), a view (`data.values()`) or a generator; no copy is
   needed.
3. A container with no child returns an empty iterable: it is not a leaf and yields nothing. A leaf
   returns `None`: the engine yields the value itself. "Empty" and "leaf" are two different answers.
4. The children are the values to visit; `find_children` never returns the keys of a mapping.
5. `find_items` returns the same children with a label each (the key of a mapping, the index of a
   sequence, the name of a field, the element itself for a set). The default numbers the children
   returned by `find_children`, so a finder that only implements `find_children` works with
   `dfs_items` and gets integer labels. The four built-in finders override it.

### The engines

```python
def depth_first(data, registry):
    find = _make_find(registry)  # closure over a per-call dict: type -> finder.find_children
    children = find(data)
    if children is None:
        yield data
        return
    ancestors = {id(data)}  # ids of the containers on the path
    stack = [(data, iter(children))]  # the container is kept: its id cannot be reused
    while stack:
        owner, it = stack[-1]
        for item in it:  # resumes where it stopped
            children = find(item)
            if children is None:
                yield item
                continue
            if id(item) in ancestors:
                raise ValueError(f"Cycle detected: an object of type {type(item)} contains itself")
            ancestors.add(id(item))
            stack.append((item, iter(children)))
            break
        else:  # the children are exhausted
            stack.pop()
            ancestors.discard(id(owner))


def breadth_first(data, registry):
    find = _make_find(registry)
    queue = deque([(data, None)])  # (item, chain of its ancestors: (container, parent chain))
    expanded = set()  # ids already expanded: a cycle needs one of them twice
    while queue:
        item, chain = queue.popleft()
        children = find(item)
        if children is None:
            yield item
            continue
        if id(item) in expanded:  # rare: walk the chain, compare with ``is``
            node = chain
            while node is not None:
                if node[0] is item:
                    raise ValueError(
                        f"Cycle detected: an object of type {type(item)} contains itself"
                    )
                node = node[1]
        expanded.add(id(item))
        chain = (item, chain)
        queue.extend((child, chain) for child in children)
```

Design points:

- **No `RecursionError`, no thread-local, no lock.** The state (stack or queue, cache, ancestors) is
  local to the call, so threads and re-entrant calls (a consumer that starts another traversal) are
  independent. The registry is not copied or wrapped for a traversal.
- **The objects, not their ids, are kept on the path** (the stack of DFS, the chain of BFS), so an
  id cannot be reused by a temporary while it is an ancestor. BFS keeps the `expanded` set of
  ids as a cheap filter: a false positive (a freed object whose id is reused) costs one walk of the
  chain and ends with `is`, never with an error. The walk is linear only for a container met
  twice, so a deep chain stays linear in its depth.
- **The cycle check is after the leaf rule**, so a leaf (an iterator, a string) never enters the
  sets, and a container is reported even if a custom finder returns its own object as a child.
- **The per-call cache** removes the registry lookup per node. `_make_find` resolves
  `registry.find_child_finder(type(x)).find_children` once per type and per call; a registration
  made during a traversal is seen by the next call.
- **A cycle raises when it is reached**, not before: the traversal is lazy, so the leaves before it
  were already yielded. Documented in the property "Cycle-safe".
- Closing the generator early (a `break`) needs no clean-up: there is no shared state to restore.

### Built-in child finders

| Child finder           | `find_children(data)`                                                               | Default registration                                                      |
| ---------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `DefaultChildFinder`   | `None` (leaf)                                                                       | `object`, `str`, `bytes`, `int`, `float`, `complex`, `bool`, `UserString` |
| `IterableChildFinder`  | `data` (the container is its own iterable of children)                              | `list`, `tuple`, `deque`, `Sequence`, `set`, `frozenset`                  |
| `MappingChildFinder`   | `data.values()`                                                                     | `dict`, `Mapping`, `MappingProxyType`                                     |
| `DataclassChildFinder` | `(getattr(data, f.name) for f in dataclasses.fields(data))`                         | fallback: `dataclasses.is_dataclass`                                      |
| `PydanticChildFinder`  | the values of `model_fields` then the extra fields; lazy: `coola.iterator.pydantic` | `pydantic.BaseModel`                                                      |

`DefaultChildFinder` keeps the explicit `str`, `bytes` and `UserString` registrations: `object` would
be enough through the MRO, but `str` and `bytes` are `Sequence`s and `UserString` would never end,
so listing them documents the types that must not be expanded and makes `has_child_finder` answer
for them. The one table of types is in `interface.py` (`_register_default_child_finders`), the only
place that says what is a container.

The extension example of the guide has no registry to remember and one registration for both orders:

```python
class NodeChildFinder(BaseChildFinder[Node]):
    def find_children(self, data):
        return [data.left, data.right]


register_child_finders({Node: NodeChildFinder()})
list(dfs_iterate(tree))
list(bfs_iterate(tree))
```

### The path of each leaf: `dfs_items`

`dfs_items(data, *, registry=None, empty=False)` yields `(path, leaf)` pairs: the leaves of
`dfs_iterate`, in the same order, each with the tuple of the labels of the containers crossed to reach
it. The root being a leaf gives the empty path. With `empty=True` an empty container is also yielded as
a leaf; by default it yields nothing.

- **The shape of a type is known in one place: its finder.** A finder gets the optional method
  `find_items(data) -> Iterable[tuple[label, child]] | None`, with the same `None` for a leaf. The
  default derives it from `find_children` (`enumerate`), and the built-in finders override it
  (`data.items()` for a mapping, `enumerate(data)` for a sequence, the field names for a dataclass,
  the element itself for a set: a set has no stable position). `find_children` stays the method of the
  hot path, so `dfs_iterate` and `bfs_iterate` pay nothing for the labels.
- **The labels are raw**: a key is not converted to `str`. They are the keys, indices and field names,
  not the path parts of `equality` (`AttributeName`, `Element`).
- **One engine, three loops.** `_make_find` is parametrised by the method name (`find_children` or
  `find_items`), so the three loops share the per-call finder cache, the cycle rule and the leaf rule.
  `depth_first_items` keeps a parallel list of labels that is appended and popped with the stack frame.
- **Depth first only.** There is no `bfs_items`: the labels of a breadth-first walk would need the
  path in every queue entry.
- **Same leaves, same order.** A property test ties `dfs_items` to `dfs_iterate`.

## Public API

```python
from coola.iterator import (
    bfs_iterate,  # (data, *, registry=None)
    dfs_iterate,  # (data, *, registry=None)
    dfs_items,  # (data, *, registry=None, empty=False)
    filter_by_type,  # (iterable, types)
    ChildFinderRegistry,
    get_default_registry,
    get_default_child_finder_registry,  # alias
    register_child_finders,  # (mapping, *, exist_ok=False)
    BaseChildFinder,
    ChildFinder,  # protocol, find_children() only
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
    DataclassChildFinder,
)
```

The package is flat: `base.py`, `default.py`, `iterable.py`, `mapping.py`, `dataclass.py`,
`registry.py`, `engine.py`, `interface.py`, `filtering.py` (the layout of `coola.recursive`).
`ChildFinderRegistry` has `register`, `register_many`, `has_child_finder`, `find_child_finder`, and the
inherited `copy`, `unregister`, `clear`. The engine is the only walker; the doctests use
`dfs_iterate(data, registry=...)`. `register_child_finders(mapping, *, exist_ok=False)` has a
keyword-only `exist_ok`. The plugin group is `coola.iterator.child_finders`, like
`coola.recursive.transformers`.

`filter_by_type` has a signature that the type checker understands (`types: type[T] | tuple[type[T],
...]`).

## Dependents

| Package or page     | Use                                                                                                                                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `coola.recursive`   | The order of the calls of `func` equals the order of `dfs_iterate` (a property test ties the two).                                                                             |
| `coola.nested`      | Built on the engines of `recursive` and `iterator`.                                                                                                                            |
| `pyproject.toml`    | `coola.iterator` is in the list of the modules that the core must not import.                                                                                                  |
| Third-party plugins | Register a `ChildFinder` in the plugin group `coola.iterator.child_finders`. A finder returns the children instead of recursing (reversing a list is `return reversed(data)`). |

## Design choices and trade-offs

**One child finder and one registry for DFS and BFS.** The two orders differ only by the order in which
the same children are visited. One `find_children` per type, walked by several engines, avoids
duplicated classes, registries, singletons and plugin groups, and makes "same leaves" a property of the
design instead of a convention. The cost is that a custom finder cannot choose its own order for one
traversal only: it returns its children in the order they are visited, for both orders. Two packages
with two registries would be cheaper to start, but every feature (a third order, a depth limit, the
paths) would be written twice.

**Two functions, not one with an `order` argument.** `dfs_iterate(data)` and `bfs_iterate(data)` read
well, appear in the guide and in the `recursive` property ("the order of `dfs_iterate`"), and an
`order="dfs"` string would add an option that needs validation.

**The containers entered by default are those of `recursive`** (`list`, `tuple`, `deque`, `Sequence`,
`set`, `frozenset`, `dict`, `Mapping`, `MappingProxyType`, the dataclasses), so that the same data
gives the same leaves in `iterator`, `recursive`, `hashing` and `summary`. The consequence is that
`range(3)` is a leaf (it is a lazy arithmetic sequence, like an array) and that a user class that only
inherits `Collection` or `collections.abc.Set` is a leaf until it registers a finder. Widening
`recursive` to `Collection` is not possible, because `recursive` must rebuild the type.

**The plugin group is `coola.iterator.child_finders`**, the same form as `coola.recursive.transformers`,
`coola.equality.testers` and `coola.hashing.hashers`.

**`filter_by_type` is kept.** It is three lines, documented and tested. The alternative is a generator
expression `(x for x in dfs_iterate(data) if isinstance(x, int))`, which the type checker types less
well.

**`ChildFinder` and not `Expander`.** The method is `find_children` and the registry method is
`find_child_finder`. `recursive` uses `expand` because it returns the children and the rebuild
function; the two names differ because the two contracts differ.

**The registry is not shared with `recursive`.** A `Transformer.expand` returns the children and a
rebuild function; the iterator needs only the children. One registry for both would make `dfs_iterate`
build a closure per container and couple two packages that have different extension points (a custom
type that can be traversed but not rebuilt is a normal case). `find_children` is a subset of `expand`,
so a merge stays possible if a third traversal package wants the same children.

**Shared containers are not memoized.** A container reachable by several paths gives its leaves once
per path (`[s, s]` gives them twice). This is the property "Leaves only, once per occurrence", and
`recursive_apply` calls `func` the same number of times. A DAG with millions of paths is slow; that is
documented, not hidden.

**No `registry.iterate` method.** `recursive` has `TransformerRegistry.transform` as the registry-level
entry point; here the registry is a dispatcher and the functions are the only entry points, so there is
one way to traverse and no method that has to choose an order.

## Not covered

- Paths in breadth-first order: `dfs_items` is the only function that yields paths.
- Pydantic models, attrs classes and plain objects with `__dict__`: leaves. `equality`
  enters pydantic models; a lazily registered `PydanticChildFinder` is a separate item, with the same
  one for `recursive`.
- NumPy arrays, torch tensors, data frames: leaves (the function receives the whole array).
  Element-wise iteration is the job of the caller.
- Parallel or async traversal, and a `max_depth` or `max_leaves` option: out of scope; `itertools.islice`
  stops a traversal after `n` leaves, because the functions are lazy.
- `coola.recursive`: the engine is separate (it must rebuild) and already iterative; only the
  container table and the order are shared.

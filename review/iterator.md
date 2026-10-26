# `coola.iterator`: target state and the changes to reach it

This document describes the design that `coola.iterator` should have, then the changes to make to
the current implementation (`main`, `8498d971`) to get there. The design follows the rules of the
user guide (`docs/docs/uguide/iterator.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: registry dispatch, MRO lookup, lazy imports, strict contracts), and
it stays aligned with the sibling packages `recursive`, `equality`, `hashing` and `summary`
(`review/recursive.md` is the model: **one engine, one extension point, one rule per container
type**). Its aim is a small package that is easy to maintain. Breaking changes are accepted, and the
future state is more important than the past one.

Evidence comes from probes and a throw-away prototype run on `main` (Python 3.14, scripts kept
outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from the
code. Nothing in `src`, `tests` or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 1,554 lines (and 2,400 lines of tests) for two
   functions that walk the same tree in two orders. The target removes the code that exists only
   because the two orders were built as two parallel packages (two registries, two sets of
   handlers, two default singletons, two plugin groups, the `bootstrap` module that keeps the two
   in sync) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. `coola.iterator.dfs` and `coola.iterator.bfs` disappear, the
   deprecated positional `registry` is removed. The release notes and the migration page list the
   changes.
3. **No slower than today.** `dfs_iterate` and `bfs_iterate` are the primitives of the traversal
   helpers and of the documentation examples. The benchmark gate of section 8 must hold; a design
   that is simpler but slower is rejected.
4. **Consistent with the siblings.** The same input must be handled the same way by `equality`,
   `hashing`, `summary`, `recursive` and `iterator`: the same containers are entered, dataclasses
   are entered, iterators are leaves, cycles are detected, options are keyword-only, and the order
   of `dfs_iterate` is the order in which `recursive_apply` calls `func`.
5. **Optional libraries stay lazy.** The package imports no optional library; NumPy arrays, pandas
   frames, ... are leaves unless a child finder is registered (the plugin group is the extension
   point).

## 1. Properties of the sub-package

These are the rules of the user guide, made complete. Every design choice below exists to make one
of them true **by construction** (in the engine) instead of by the discipline of each handler.

| Property                 | Statement                                                                                                                                                                                                                                                      | Enforced by                       |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Leaves only              | A traversal yields the leaves, exactly once per occurrence, never a container. A leaf is a value whose child finder returns `None` (scalars, `str`, `bytes`, iterators, arrays, ...). An empty container has no leaf and yields nothing.                       | the engine                        |
| Type dispatch            | The child finder is chosen from `type(data)` along the Method Resolution Order, so the most specific registered type wins. A finder is always found: `object` is registered by the default registry.                                                           | the registry                      |
| Registry-extensible      | A new type is supported by registering one child finder, without changing another one and without subclassing a base class (a protocol is enough). One registration serves both orders.                                                                        | the registry, the protocol        |
| Deterministic order      | DFS is pre-order, left to right. BFS is level order, left to right inside a level. A mapping gives its values in insertion order (keys are never yielded), a dataclass its fields in the order of their definition, a set its iteration order (not specified). | the engine, one rule per finder   |
| Same leaves              | `dfs_iterate` and `bfs_iterate` yield the same leaves (the same multiset); only the order differs. A custom finder cannot break it: both orders call the same finder.                                                                                          | one registry, one finder per type |
| Lazy                     | Both functions are generators. The leaves are produced on demand, an early `break` stops the traversal, and nothing is walked before the first `next`. The memory is O(depth) for DFS and O(width of a level) for BFS.                                         | generators                        |
| Read-only                | The input is never modified. An iterator or a generator found in the data is a leaf: it is returned as it is and never consumed, so an infinite iterator cannot hang the traversal.                                                                            | the default registry              |
| Cycle-safe               | A container that contains itself raises `ValueError` when the traversal reaches it (the leaves before it are already yielded), whichever finder is registered for it. A container reachable by several paths without a cycle is not an error.                  | the engine                        |
| Any depth                | No `RecursionError`, whatever the nesting or the caller's stack: both engines use an explicit stack or queue. The depth is bounded by memory only.                                                                                                             | the engine                        |
| Errors propagate         | An exception of a finder, or of the data (for example a failing property), propagates unchanged. An unregistered type raises `TypeNotRegisteredError` (only for a registry without `object`).                                                                  | the engine                        |
| Consistent with siblings | Dataclasses are entered (like `objects_are_equal`, `hash_object`, `summarize`, `recursive_apply`); the containers are the ones of `recursive`; options are keyword-only; the DFS order is the order of the calls of `recursive_apply`.                         | the default registry              |
| Fast                     | The cost is linear in the number of nodes; the finder lookup is cached per call; no lock, no thread-local, no registry object created per call.                                                                                                                | the engine                        |
| Thread-safe              | The state of a traversal (stack or queue, cache, ancestors) is local to the call. A registry and a finder can be shared by threads and calls; a traversal must not be shared.                                                                                  | the engine, stateless finders     |

Not guaranteed, and documented as such:

- **Shared references.** A container reachable by several paths (a DAG) gives its leaves once per
  path, so the cost is proportional to the number of paths, not to the number of objects. No
  memoization: it would break the property "Leaves only, once per occurrence" (decision D8).
- **The order inside a set** and the order of a `ChainMap` (its `values()` follow the lookup order of
  the maps, last map first **[verified]**: `dfs_iterate(ChainMap({"a": 1}, {"b": 2}))` gives `[2, 1]`).
- **Mutation during the traversal.** Changing a container while it is walked is the caller's
  problem, as for any Python iteration (`RuntimeError` for a dict or a set that changes size).
- **A registry modified during a traversal.** The traversal keeps the finders it already resolved.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Status         |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **Two packages for one concept.** `dfs` and `bfs` each have a base class, a default, an iterable, a mapping and a dataclass handler (10 classes for 5 ideas), a registry, a default singleton, an interface module and a plugin group; `bootstrap.py` and its `SupportsRegisterMany` protocol exist only to keep the two type tables identical. A type needs two registrations (`register_iterators` and `register_child_finders`) to be traversed in both orders, and a plugin must ship two entry points. 1,403 of the 1,554 lines are in the two trees.                                                                                                                              | **[reading]**  |
| F2  | **The recursion lives in each DFS iterator.** `IterableIterator` and `MappingIterator` call `registry.iterate` for each child. The cycle check (`PathRegistry`) and the `RecursionError` translation work only if the iterator cooperates: a custom iterator that walks its children itself bypasses both. `PathRegistry` is exported in `__all__`, subclasses `IteratorRegistry`, copies the private `_state` of the registry and is built (with its own empty lookup cache) for each traversal.                                                                                                                                                                                       | **[reading]**  |
| F3  | **DFS has a depth limit.** `dfs_iterate` raises `RecursionError` at 1,000 levels; `bfs_iterate` handles 20,000 in 19 ms. The two functions of one package do not accept the same inputs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | **[verified]** |
| F4  | **DFS cost per leaf grows with the depth.** Each leaf is yielded through a chain of `yield from` generators, one per level, so a leaf at depth 250 costs 4.95 µs where a leaf at depth 1 costs 0.61 µs (8x). BFS is flat (0.51 to 0.57 µs).                                                                                                                                                                                                                                                                                                                                                                                                                                             | **[verified]** |
| F5  | **Slow in both orders.** 100,000 ints in a flat list: `dfs_iterate` 58.7 ms, `bfs_iterate` 50.1 ms, about 0.5 µs per leaf, 70 % of it registry lookups (`find_*` is called for every node, nothing is cached per call) and generator frames (two per leaf in DFS). The prototype of section 6 does the same in 9.3 ms (DFS) and 19.1 ms (BFS).                                                                                                                                                                                                                                                                                                                                          | **[verified]** |
| F6  | **BFS reports a cycle that does not exist.** The ancestors are kept as ids; the queue holds the children but not the expanded parent, so a finder that builds fresh objects (a lazy node, a property that returns a new object) frees a parent whose id is then reused by a later child. `bfs_iterate(Node(50), registry=...)` with a finder that yields `Node(n.d - 1)` raises `ValueError: Cycle detected` on a chain of 50 distinct nodes; the same finder through DFS works. The prototype keeps the objects in the ancestor chain and compares with `is`.                                                                                                                          | **[verified]** |
| F7  | **A leaf takes two members in BFS and a convention in DFS.** A BFS finder for a leaf must override the property `is_leaf` and `find_children`; if it forgets `is_leaf`, the value is "treated as an empty container" and silently disappears from the output. `DefaultChildFinder.find_children` is `return` followed by an unreachable `yield` (the generator hack) with `# noqa: ARG002`. In DFS a leaf is an iterator that yields its data (`# noqa: ARG002` for the unused registry).                                                                                                                                                                                               | **[reading]**  |
| F8  | **The containers differ from `recursive`.** The default tables of `iterator` and `recursive` are not the same. `range`, a user `Collection` and a user `collections.abc.Set` subclass are entered by `dfs_iterate` and are leaves for `recursive_apply`; `iterator` has the catch-all `Collection`, `recursive` has `Sequence`. The same data gives a different leaf set in the two packages, which breaks "the DFS order is the order of the calls of `recursive_apply`" (`range(3)`: `[0, 1, 2]` against `[range(0, 3)]`).                                                                                                                                                            | **[verified]** |
| F9  | **Deprecated and positional options kept.** `dfs_iterate(data, registry)` and `bfs_iterate(data, registry)` still go through `deprecate_positional_options` and emit a `DeprecationWarning`; `register_iterators` and `register_child_finders` take `exist_ok` positionally (the rule of `CLAUDE.md` is keyword-only).                                                                                                                                                                                                                                                                                                                                                                  | **[reading]**  |
| F10 | **`filter_by_type` is mistyped for a tuple.** `types: type[T] \| tuple[type, ...]` and `Iterator[T]` make `filter_by_type(xs, (int, str))` an `Iterator[Unknown]` for the type checker (`Iterator[int]` for a single type). The function is unrelated to the traversal (a filter on any iterable) but lives in the package, and its guide section is longer than its code.                                                                                                                                                                                                                                                                                                              | **[verified]** |
| F11 | **Other suppressions.** Two `# noqa: ARG002` (F7) and one `# pragma: no cover` (the protocol body in `bootstrap.py`). `review/noqa-review.md` asks for none of them.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | **[reading]**  |
| F12 | **The guide contradicts the code or teaches the wrong thing.** "Generator-based: memory-efficient iteration without loading entire structures" (BFS holds a whole level in a queue); "Note: dict order may vary in Python < 3.7" (3.11 is the minimum); the section "Writing an Iterator" teaches the recursion of F2; the `IterableIterator` docstring shows `"hello"` iterated into characters (a registry with only `list` and `object` and a direct call, not what `dfs_iterate` does); the `dfs_iterate` docstring has a 130-character line and says "a few hundred levels"; there is no table of the leaves of a type, and the properties omit order, laziness and thread-safety. | **[reading]**  |
| F13 | **Tests duplicate each other.** 2,400 lines for 1,554 lines of code: `tests/unit/iterator/{bfs,dfs}` and `tests/integration/iterator/{bfs,dfs}` repeat the same cases with two `helpers.py`. Nothing runs the two orders against one table of inputs, and the iterators have no benchmark (`tests/benchmarks/` covers `equality`, `recursive` and the registry only).                                                                                                                                                                                                                                                                                                                   | **[reading]**  |

What is already right and must not regress: type dispatch with MRO; `str`, `bytes` and `UserString`
as leaves (`UserString` is a `Sequence` of `UserString`: expanding it never ends); iterators and
generators as leaves; keys never yielded; a dataclass entered through `DataclassMarker`, with a
finder registered for the type having priority; cycles detected; a shared container (no cycle)
giving its leaves each time; the input unmodified; keyword-only `registry`; lazy generators; the
doctests of the guide.

## 3. Architecture

```
dfs_iterate(data, *, registry=None)          bfs_iterate(data, *, registry=None)
        |   one registry (default: the singleton), one call = one engine run
        v
engine.depth_first(data, registry)           engine.breadth_first(data, registry)
        |   coola/iterator/engine.py (about 60 lines for both)
        |   explicit stack / queue; per-call type -> finder cache; the containers on the path
        |
        |-- finder   = registry.find_child_finder(type(item))      MRO, DataclassMarker
        |-- children = finder.find_children(item)                  pure: no registry, no recursion
        |        None      -> leaf: yield item
        |        iterable  -> container: walk the children (DFS: push; BFS: enqueue)
        v
the leaves, one at a time
```

The engine owns the invariants (dispatch cache, leaf rule, cycles, depth, order, laziness). A child
finder **only describes the shape of a type**: its children. That is the same split as `recursive`
(`expand`) and `equality` (a tester returns the child pairs). The registry is no longer an argument
of the finder, so a finder cannot forget a protection and cannot recurse. Both orders call the same
finder, so "same leaves" is true by construction.

### 3.1 The child finder contract

```python
class ChildFinder(Protocol[T_contra]):
    def find_children(self, data: T_contra) -> Iterable[Any] | None:
        """``None`` for a leaf; otherwise the children of ``data`` in the order they are visited."""


class BaseChildFinder(ABC, Generic[T]):  # convenience base class, as today
    @abstractmethod
    def find_children(self, data: T) -> Iterable[Any] | None: ...
```

Rules of the contract (documented in `BaseChildFinder`, checked by one parametrised unit test over
the default registry, like `NATIVE_SAMPLES` in `equality`):

1. `find_children` is pure and cheap: it does not call the registry, does not recurse, does not
   modify `data` and does not consume the children (it returns them).
2. The result is consumed once, lazily, in order. A finder can return the container itself
   (`IterableChildFinder` returns `data`), a view (`data.values()`) or a generator; no copy is
   needed.
3. A container with no child returns an empty iterable: it is not a leaf and yields nothing. A leaf
   returns `None`: the engine yields the value itself. "Empty" and "leaf" are two different answers
   (today `is_leaf` and an unreachable `yield` encode the same distinction).
4. The children are the values to visit; a finder never returns the keys of a mapping.

### 3.2 The engines

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

- **No `RecursionError`, no `PathRegistry`, no thread-local, no lock.** The state (stack or queue,
  cache, ancestors) is local to the call, so threads and re-entrant calls (a consumer that starts
  another traversal) are independent. `raise_recursion_limit_error` is not used by this package any
  more. The registry is not copied or wrapped for a traversal.
- **The objects, not their ids, are kept on the path** (the stack of DFS, the chain of BFS), so an
  id cannot be reused by a temporary while it is an ancestor (F6). BFS keeps the `expanded` set of
  ids as a cheap filter: a false positive (a freed object whose id is reused) costs one walk of the
  chain and ends with `is`, never with an error. The walk is linear only for a container met
  twice, so a deep chain stays linear in its depth.
- **The cycle check is after the leaf rule**, so a leaf (an iterator, a string) never enters the
  sets, and a container is reported even if a custom finder returns its own object as a child.
- **The per-call cache** removes the registry lookup per node (F5). `_make_find` resolves
  `registry.find_child_finder(type(x)).find_children` once per type and per call; a registration
  made during a traversal is seen by the next call.
- **A cycle raises when it is reached**, not before: the traversal is lazy, so the leaves before it
  were already yielded. Documented in the property "Cycle-safe".
- Closing the generator early (a `break`) needs no clean-up: there is no shared state to restore
  (today `PathRegistry.iterate` needs a `finally`).

### 3.3 Built-in child finders

| Child finder           | `find_children(data)`                                       | Default registration                                                                            |
| ---------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `DefaultChildFinder`   | `None` (leaf)                                               | `object`, `str`, `bytes`, `int`, `float`, `complex`, `bool`, `UserString`                       |
| `IterableChildFinder`  | `data` (the container is its own iterable of children)      | `list`, `tuple`, `deque`, `Sequence`, `set`, `frozenset` (decision D3: the list of `recursive`) |
| `MappingChildFinder`   | `data.values()`                                             | `dict`, `Mapping`, `MappingProxyType`                                                           |
| `DataclassChildFinder` | `(getattr(data, f.name) for f in dataclasses.fields(data))` | `DataclassMarker`                                                                               |

`DefaultChildFinder` keeps the explicit `str`, `bytes` and `UserString` registrations: `object` would
be enough through the MRO, but `str` and `bytes` are `Sequence`s and `UserString` would never end,
so listing them documents the types that must not be expanded and makes `has_child_finder` answer
for them. The one table of types moves from `bootstrap.py` into `interface.py`
(`_register_default_child_finders`), where it is the only place that says what is a container.

Removed: `DfsIterator` family (`BaseIterator`, `DefaultIterator`, `IterableIterator`,
`MappingIterator`, `DataclassIterator`), `IteratorRegistry`, `PathRegistry`, `is_leaf`,
`registry.iterate`, `registry.find_children`, `bootstrap.py`, `SupportsRegisterMany`.

The extension example of the guide becomes the shortest possible one, with no registry to remember
and one registration for both orders:

```python
class NodeChildFinder(BaseChildFinder[Node]):
    def find_children(self, data):
        return [data.left, data.right]


register_child_finders({Node: NodeChildFinder()})
list(dfs_iterate(tree))
list(bfs_iterate(tree))
```

## 4. Public API after the change

```python
from coola.iterator import (
    bfs_iterate,  # (data, *, registry=None)
    dfs_iterate,  # (data, *, registry=None)
    filter_by_type,  # (iterable, types)
    ChildFinderRegistry,
    get_default_registry,
    get_default_child_finder_registry,  # alias, unchanged
    register_child_finders,  # (mapping, *, exist_ok=False)
    BaseChildFinder,
    ChildFinder,  # protocol, find_children() only
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
    DataclassChildFinder,
)
```

`coola.iterator.dfs` and `coola.iterator.bfs` are deleted (no alias): there is nothing left in
them that differs. The flat package is `base.py`, `default.py`, `iterable.py`, `mapping.py`,
`dataclass.py`, `registry.py`, `engine.py`, `interface.py`, `filtering.py` (the layout of
`coola.recursive`). `ChildFinderRegistry` keeps `register`, `register_many`, `has_child_finder`,
`find_child_finder`, and the inherited `copy`, `unregister`, `clear`; it loses `iterate` and
`find_children` (the engine is the only walker; the doctests use `dfs_iterate(data, registry=...)`).
`register_child_finders(mapping, *, exist_ok=False)` becomes keyword-only for `exist_ok`
(`CLAUDE.md`). The plugin groups `coola.iterator.dfs` and `coola.iterator.bfs` are replaced by one
group, `coola.iterator.child_finders`, like `coola.recursive.transformers` (decision D4).

`filter_by_type` stays, with a signature that the type checker understands
(`types: type[T] | tuple[type[T], ...]`, decision D5).

## 5. Dependents

| Package or page                                                                                                                                                                                                                                                  | Change                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.recursive`                                                                                                                                                                                                                                                | No code change. The order of the calls of `func` must equal the order of `dfs_iterate` (the property test of section 8 ties the two); D3 aligns the containers.                                                                           |
| `coola.nested`, `coola.rounding`                                                                                                                                                                                                                                 | No import of `coola.iterator` in `src` (verified by `grep`); nothing to change.                                                                                                                                                           |
| `pyproject.toml` (`forbidden_modules` list)                                                                                                                                                                                                                      | `coola.iterator` stays in the list of the modules that the core must not import; no change.                                                                                                                                               |
| `tests/package_checks.py`, `tests/unit/test_public_api.py`, `tests/unit/test_default_registries.py`, `tests/unit/registry/test_dispatch.py`, `tests/integration/test_dataclasses.py`, `tests/properties/test_traversal.py`, `tests/properties/test_recursive.py` | Import the flat names (`from coola.iterator import ChildFinderRegistry, ...`); the test of the default registries has one registry to check instead of two.                                                                               |
| `docs/docs/refs/iterator.md`                                                                                                                                                                                                                                     | One `::: coola.iterator` entry; the `coola.iterator.bfs` and `coola.iterator.dfs` entries are deleted.                                                                                                                                    |
| `docs/docs/uguide/iterator.md`                                                                                                                                                                                                                                   | Rewritten (step 7): the **Properties** table first, the finder example, the container table, no `IteratorRegistry`.                                                                                                                       |
| `docs/docs/uguide/extending.md`                                                                                                                                                                                                                                  | The two plugin rows become `coola.iterator.child_finders`.                                                                                                                                                                                |
| `docs/docs/uguide/registry.md`, `recursive.md`, `nested.md`, `dev/add_backend.md`, `dev/architecture.md`                                                                                                                                                         | Update the names of the registry and the links; `add_backend.md` already says the objects are leaves for `iterator`.                                                                                                                      |
| `docs/docs/migration/v1.1_to_v2.0.md`                                                                                                                                                                                                                            | A section "Iterators: one child finder for both orders" (before/after table, the plugin group, the removed names).                                                                                                                        |
| Third-party plugins of `coola.iterator.dfs` / `.bfs`                                                                                                                                                                                                             | Must register a `ChildFinder` in `coola.iterator.child_finders`. A DFS `BaseIterator` becomes a `find_children` that returns the children instead of recursing (the `ReversedListIterator` of the guide becomes `return reversed(data)`). |

## 6. Evidence: a prototype of the engines

A prototype of the two loops above (explicit stack and queue, the real default registry for the
lookup with a per-call cache, `find_children` inlined as a `None`-or-generator) was run on `main`. It
is not part of the repository. **[verified]**

| Input                                                       | `dfs_iterate` on `main`  | DFS prototype | `bfs_iterate` on `main` | BFS prototype |
| ----------------------------------------------------------- | ------------------------ | ------------- | ----------------------- | ------------- |
| flat list of 100,000 ints                                   | 58.7 ms                  | 9.3 ms        | 50.1 ms                 | 19.1 ms       |
| tree of 10,000 leaves (100 x 10 x 10 lists)                 | 6.84 ms                  | 1.35 ms       | 5.85 ms                 | 2.49 ms       |
| list nested 50,000 levels deep                              | `RecursionError` (1,000) | works         | works (20,000 in 19 ms) | works         |
| a cycle `[1, cycle]`                                        | `ValueError`             | `ValueError`  | `ValueError`            | `ValueError`  |
| leaf cost, depth 1 / 250 (2,000 leaves)                     | 0.61 / 4.95 µs           | flat          | 0.51 / 0.57 µs          | flat          |
| chain of 50 temporary nodes (F6)                            | works                    | works         | false `ValueError`      | works         |
| output equals the output of `main` on `[1, 2, [3, [4]], 5]` | n/a                      | yes           | n/a                     | yes           |

DFS is 4 to 6 times faster because the `yield from` chain, the two generator frames per leaf and the
per-node registry lookup are gone; BFS is 2 to 2.6 times faster for the lookup and the single
`deque.extend` of a generator. The BFS prototype is slower than the DFS one because every child is
queued with its ancestor chain (one tuple per node); the memory is O(width of a level). The real
engine calls a method per node and may be a little slower than the prototype, so the gate (section 8) is "no slower than `main`", with the margin above.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                  | Breaking                           |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 1    | **Fix the guide (F12) and add benchmarks.** Remove "memory-efficient", the Python 3.7 note, fix the docstrings, add the **Properties** table of section 1 at the top of the page and in the package docstring. Add `tests/benchmarks/test_iterator_benchmark.py` (the inputs of section 6) and save the baseline of `main`. Documentation and tests only.                                                               | no                                 |
| 2    | **One iterator test table.** A parametrised property and integration test that runs `dfs_iterate` and `bfs_iterate` on one table of inputs (leaves, empty containers, dataclass, generator, cycle, shared container), and the cross-package test "same containers as `recursive`" (F8). They pass or fail on the old code, then guard the rewrite.                                                                      | no                                 |
| 3    | **Engines and the `find_children` protocol (F1 to F7).** Add `engine.py` and the flat modules, one registry (`ChildFinderRegistry`), the `ChildFinder` protocol and the four finders, one `register_child_finders`. Delete `coola.iterator.dfs`, `coola.iterator.bfs`, `bootstrap.py`, `PathRegistry`, `is_leaf` and the suppressions. Merge the unit tests of the two trees into one suite per module. Benchmark gate. | yes (names, plugins)               |
| 4    | **Align the containers with `recursive` (F8, D3).** The default table is `Sequence`, `set`, `frozenset`, `Mapping`, ... as `recursive`; `range` and the catch-all `Collection` are removed. Can be merged into step 3.                                                                                                                                                                                                  | yes (`range`, custom `Collection`) |
| 5    | **Remove the deprecated and positional paths (F9).** The positional `registry` of `dfs_iterate` and `bfs_iterate`, `exist_ok` keyword-only. Can be merged into step 3.                                                                                                                                                                                                                                                  | yes                                |
| 6    | **`filter_by_type` (F10).** The typed signature and a test for the tuple case (checked with `ty`); shorten the guide section.                                                                                                                                                                                                                                                                                           | no                                 |
| 7    | **Documentation.** Rewrite `uguide/iterator.md` (finder example, container table, properties, DAG and set-order notes), the plugin table of `uguide/extending.md`, `refs/iterator.md`, the other links of section 5, the migration page, and the status table of `review/suggestions.md` (items 25.1 `dfs_iterate`, 28.2 with the PR numbers).                                                                          | no                                 |

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/iterator/`, one file per module, no `dfs` and `bfs`
sub-directories); the conventions of `CLAUDE.md` apply (one behaviour per test, literal expected
values, a regression test for every bug, no local imports, section headers).

- **Engine** (`test_engine.py`, new, parametrised over `depth_first` and `breadth_first` where the
  behaviour is the same, so a rule is written once):
  - the order on a nested list/dict/tuple/set structure: DFS and BFS asserted against a full literal
    (`[1, 2, 3, 4, 5]` against `[1, 4, 2, 5, 3]` for `[1, [2, [3]], 4, [5]]`);
  - an empty container yields nothing; a root leaf yields itself; `None` is a leaf;
  - a shared container (`[s, s]`) gives its leaves twice;
  - a cycle through a list, a dict and a tuple-in-list raises `ValueError`, whichever finder is
    registered (a custom finder that returns its own object as a child);
  - the leaves before a cycle are yielded before the error (laziness);
  - depth 100,000 does not raise, in both orders (regression for F3; check it fails on `main` with
    `git stash push src`);
  - **regression for F6**: a chain of 50 temporary nodes does not raise in BFS (fails on `main`);
  - an exception of a finder propagates unchanged and leaves no state: the next call works;
  - early `break` then a new traversal works; two interleaved traversals of the same registry are
    independent;
  - the per-call cache: a finder registered during a traversal is not used by this call;
  - `TypeNotRegisteredError` for an empty registry (as today).
- **Each finder** (`test_<module>.py`): `find_children` asserts the whole output against a literal
  (typical, empty, single element, duplicates), and `None` for a leaf; subclasses of `list`, `dict`
  and `tuple`; named tuple, `deque`, `OrderedDict`, `Counter`, `MappingProxyType`; the input is not
  modified. Dataclass: frozen, slots, `init=False` field, nested dataclass, a finder registered for
  the type has priority over the marker, `DefaultChildFinder` keeps a dataclass as a leaf.
- **Contract test over the default registry**: every finder of the default registry returns `None`
  or an iterable whose leaves equal the leaves of the sample (one sample per registered type, like
  `NATIVE_SAMPLES`); the finder is pure (two calls give the same children).
- **Registry** (`test_registry.py`): MRO lookup, `has_child_finder`, `find_child_finder` for a
  dataclass, `copy`, repr.
- **Interface** (`test_interface.py`): `registry=None` uses the default, a custom registry is used,
  `registry` is keyword-only (`TypeError` otherwise, which replaces the deprecation test),
  `register_child_finders(..., exist_ok=...)`, `exist_ok` keyword-only, the aliases.
- **Properties** (`tests/properties/test_traversal.py`, extend): the leaves of each order equal the
  leaves of the generated data (kept); same leaves in both orders (kept); the order of a list is the
  order of the elements (kept); the input is not modified (kept); **the order of the calls of
  `func` in `recursive_apply` equals the order of `dfs_iterate`** (new, ties `recursive` and
  `iterator`, guards F8); **the leaf set is the same for `iterator` and `recursive` on a table of
  container types** (new).
- **Integration** (`tests/integration/iterator/`): threads running both functions on the default
  registry with a lock around the shared list (`run_threads`, docstring "Test that ...");
  a plugin entry point registered through `coola.iterator.child_finders` (no patching);
  a custom finder that serves both orders; NumPy and pandas objects are leaves (using the
  `*_available` fixtures); no import of an optional library by `import coola.iterator`.
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Benchmark gate** (`tests/benchmarks/test_iterator_benchmark.py`, the CI benchmark): flat list,
  tree, deep list (10,000 levels, not possible on `main` for DFS), list of small dicts and mixed
  containers, for both functions; the new engines must not be slower than `main` on any input that
  `main` can run (expected: 2 to 6 times faster). A failing gate rejects the step.
- Coverage: 100 % of `src/coola/iterator` (`--cov=coola`), no `# pragma: no cover`.

## 9. Decisions for the maintainers

**D1. One child finder and one registry for DFS and BFS?** _Recommended: yes._ The two packages
differ only by the order in which the same children are visited. One `find_children` per type,
walked by two engines, removes the duplicated classes (10 for 5 ideas), the duplicated registries,
singletons and plugin groups, the `bootstrap` module, and makes "same leaves" a property of the
design instead of a convention. The cost is a breaking rename and a custom DFS iterator that chose
its own order (`ReversedListIterator`) can no longer do it for DFS only: it returns its children in
the order they are visited, for both orders. Alternative: keep two packages and two registries;
cheaper for the release, but every future feature (a third order, a depth limit, the paths) is
written twice.

**D2. Two functions or one with an `order` argument?** _Recommended: two functions._
`dfs_iterate(data)` and `bfs_iterate(data)` read well, are in the guide and in `recursive`'s
property ("the order of `dfs_iterate`"), and an `order="dfs"` string would add an option that needs
validation. Both call one function of `engine.py` each.

**D3. Which containers are entered by default?** _Recommended: the list of `recursive`_ (`list`,
`tuple`, `deque`, `Sequence`, `set`, `frozenset`, `dict`, `Mapping`, `MappingProxyType`, the
dataclasses), so that the same data gives the same leaves in `iterator`, `recursive`, `hashing` and
`summary` (constraint 4). The consequence is that `range(3)` becomes a leaf (it is a lazy arithmetic
sequence, like an array) and that a user class that only inherits `Collection` or `collections.abc.Set`
is a leaf until it registers a finder; the migration page says so. Alternative: widen `recursive` to
`Collection`; rejected, because `recursive` must rebuild the type and cannot for an arbitrary
`Collection`.

**D4. Name of the plugin group.** _Recommended: `coola.iterator.child_finders`_, the same form as
`coola.recursive.transformers`, `coola.equality.testers`, `coola.hashing.hashers`. The two old groups
are not read any more; the migration page lists the change. Alternative: keep reading the old
groups; rejected, because it keeps the two-registry design alive.

**D5. Keep `filter_by_type`?** _Recommended: keep, fix the type._ It is three lines, documented,
tested and used in the guide with `dfs_iterate`; removing it saves nothing and moves users to a
generator expression that the type checker types less well. Alternative: delete it and show
`(x for x in dfs_iterate(data) if isinstance(x, int))` in the guide; also defensible, and the only
cost is the migration note.

**D6. Public name `ChildFinder` or `Expander`.** _Recommended: `ChildFinder`/`BaseChildFinder`_ (the
names of the BFS package, which are the better ones: the method is `find_children`, the registry
method is `find_child_finder`). `recursive` uses `expand` because it returns the children and the
rebuild function; the two names differ because the two contracts differ.

**D7. Share the registry with `recursive`?** _Recommended: not now._ A `Transformer.expand` returns
the children and a rebuild function; the iterator needs only the children. One registry for both
would make `dfs_iterate` build a closure per container and couple two packages that have different
extension points (a custom type that can be traversed but not rebuilt is a normal case). Revisit if
a third traversal package (`summary`, `hashing`) wants the same children; the contract
`find_children` is a subset of `expand`, so the merge stays possible.

**D8. Memoize the shared containers?** _Recommended: no._ A container reachable by several paths
gives its leaves once per path (`[s, s]` gives them twice). This is the property "Leaves only,
once per occurrence", and `recursive_apply` calls `func` the same number of times. Cost: a DAG with
millions of paths is slow; that is documented, not hidden.

**D9. Yield the path of each leaf?** _Recommended: not now._ A `dfs_items(data)` that yields
`(path, leaf)` would answer "where is the NaN" and is the natural next feature, but it needs a label
for each child (`find_children` would return `(label, child)` pairs, a tuple per child on the hot
path). The contract can be extended without changing the structure of the engines; revisit if users
ask. Same decision as D4 of `recursive.md`.

**D10. `registry.iterate` kept as a method?** _Recommended: removed._ `recursive` keeps
`TransformerRegistry.transform` as the registry-level entry point; here the registry is a dispatcher
and the functions `dfs_iterate` and `bfs_iterate` are the only entry points, so there is one way to
traverse and no method that has to choose an order.

## 10. Not covered

- Keys: no iterator over the keys or the items of a mapping. `dfs_items(data)` with paths (D9) is
  the natural next function once the engines exist.
- Pydantic models, attrs classes and plain objects with `__dict__`: leaves, as today. `equality`
  enters pydantic models; a lazily registered `PydanticChildFinder` is a separate item, with the same
  one for `recursive`.
- NumPy arrays, torch tensors, data frames: leaves (the function receives the whole array).
  Element-wise iteration is the job of the caller.
- Parallel or async traversal, and a `max_depth` or `max_leaves` option: out of scope; `itertools.islice`
  stops a traversal after `n` leaves, because the functions are lazy.
- `coola.recursive`: the engine is separate (it must rebuild) and already iterative; only the
  container table and the order are shared.
- The existing `suggestions.md` item 25.1 for `dfs_iterate` (explicit stack) is closed by step 3.

## Appendix: effect on the size of the package

| Module                                                    | Lines today | Expected after | Why                                                                  |
| --------------------------------------------------------- | ----------- | -------------- | -------------------------------------------------------------------- |
| `bfs/` (8 modules)                                        | 695         | 0              | merged                                                               |
| `dfs/` (8 modules)                                        | 708         | 0              | merged                                                               |
| `bootstrap.py`                                            | 76          | 0              | the table moves to `interface.py`                                    |
| `base.py`                                                 | 0           | about 90       | protocol, base class, the contract (the docstrings are the bulk)     |
| `default.py`, `iterable.py`, `mapping.py`, `dataclass.py` | 0           | about 150      | four finders of 3 to 8 lines of code each, with their doctests       |
| `registry.py`                                             | 0           | about 100      | `ChildFinderRegistry`: `has_child_finder`, `find_child_finder`       |
| `engine.py`                                               | 0           | about 90       | `depth_first`, `breadth_first`, the per-call finder cache            |
| `interface.py`                                            | 0           | about 150      | `dfs_iterate`, `bfs_iterate`, the defaults, `register_child_finders` |
| `filtering.py`, `__init__.py`                             | 75          | about 110      | the property list in the package docstring                           |
| **Package**                                               | **1,554**   | **about 690**  |                                                                      |
| Tests (unit and integration)                              | 2,400       | about 1,300    | one suite for two orders, no duplicated helpers                      |

The line counts are estimates from the prototype and from the code to delete; the gate is the
behaviour (sections 1 and 8), not the size.

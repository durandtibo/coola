# Architecture of `coola.iterator`

:book: This page describes the design of the `coola.iterator` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/iterator.md#properties); the API is in the [reference](../refs/iterator.md).

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
3. **No slower than before.** `dfs_iterate` and `bfs_iterate` are the primitives of the traversal
   helpers and of the documentation examples. The benchmark gate must hold; a design
   that is simpler but slower is rejected.
4. **Consistent with the siblings.** The same input must be handled the same way by `equality`,
   `hashing`, `summary`, `recursive` and `iterator`: the same containers are entered, dataclasses
   are entered, iterators are leaves, cycles are detected, options are keyword-only, and the order
   of `dfs_iterate` is the order in which `recursive_apply` calls `func`.
5. **Optional libraries stay lazy.** The package imports no optional library; NumPy arrays, pandas
   frames, ... are leaves unless a child finder is registered (the plugin group is the extension
   point).

## Architecture

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

### The child finder contract

```python
class ChildFinder(Protocol[T_contra]):
    def find_children(self, data: T_contra) -> Iterable[Any] | None:
        """``None`` for a leaf; otherwise the children of ``data`` in the order they are visited."""


class BaseChildFinder(ABC, Generic[T]):  # convenience base class, as before
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
   (before: `is_leaf` and an unreachable `yield` encode the same distinction).
4. The children are the values to visit; a finder never returns the keys of a mapping.

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

- **No `RecursionError`, no `PathRegistry`, no thread-local, no lock.** The state (stack or queue,
  cache, ancestors) is local to the call, so threads and re-entrant calls (a consumer that starts
  another traversal) are independent. `raise_recursion_limit_error` is not used by this package any
  more. The registry is not copied or wrapped for a traversal.
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
- Closing the generator early (a `break`) needs no clean-up: there is no shared state to restore
  (before: `PathRegistry.iterate` needs a `finally`).

### Built-in child finders

| Child finder           | `find_children(data)`                                       | Default registration                                                      |
| ---------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------- |
| `DefaultChildFinder`   | `None` (leaf)                                               | `object`, `str`, `bytes`, `int`, `float`, `complex`, `bool`, `UserString` |
| `IterableChildFinder`  | `data` (the container is its own iterable of children)      | `list`, `tuple`, `deque`, `Sequence`, `set`, `frozenset`                  |
| `MappingChildFinder`   | `data.values()`                                             | `dict`, `Mapping`, `MappingProxyType`                                     |
| `DataclassChildFinder` | `(getattr(data, f.name) for f in dataclasses.fields(data))` | `DataclassMarker`                                                         |

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

### Planned extension: the path of each leaf (`dfs_items`, `bfs_items`)

Not implemented by this work, but the design is chosen so that it is an addition, not
a rewrite:

- **The frame of the engine is one pair.** DFS keeps `(owner, iterator over the children)` per level.
  Paths add a parallel list of labels (the label of the child being visited at each level) that is
  appended and popped with the frame; BFS adds the path tuple to each queue entry. No existing line
  changes meaning.
- **The shape of a type is known in one place: its finder.** A finder gets one more, optional method,
  `find_items(data) -> Iterable[tuple[label, child]] | None`, with the same `None` for a leaf. The
  base class derives a default from `find_children` (`enumerate` of the children, so a custom finder
  works with no change and gets integer labels), and the four built-in finders override it
  (`data.items()` for a mapping, `enumerate(data)` for a sequence, the field names for a dataclass, the
  element itself for a set: a set has no stable position).
  `find_children` stays the method of the hot path, so `dfs_iterate` and `bfs_iterate` pay nothing.
- **The labels are the path parts of `equality`** (a key or an index `['a']`, `[3]`, an
  `AttributeName("x")`, an `Element(2)`), so a path printed by `compare` and a path yielded by
  `dfs_items` read the same. A path is a tuple of parts, built only by the `*_items` functions.
- **The engines are generators over a per-call finder cache** (`_make_find`). `_make_find` is
  parametrised by the method name, so the items loop reuses the cache, the cycle rule and the
  leaf rule; only the loop that carries the labels is new (about 25 lines for DFS, 15 for BFS).
- **Same leaves, same order.** `dfs_items(data)` yields the leaves of `dfs_iterate(data)` in the same
  order; a property test ties the two as soon as the function exists.

What the planned addition does **not** require: a change to the registry, to the plugin group, to
`dfs_iterate`, or to a finder written for this release.

## Public API

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
group, `coola.iterator.child_finders`, like `coola.recursive.transformers`.

`filter_by_type` stays, with a signature that the type checker understands
(`types: type[T] | tuple[type[T], ...]`, decision D5).

## Dependents

| Package or page                                                                                                                                                                                                                                                  | Change                                                                                                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.recursive`                                                                                                                                                                                                                                                | No code change. The order of the calls of `func` must equal the order of `dfs_iterate` (the property test of ties the two); D3 aligns the containers.                                                                                     |
| `coola.nested`, `coola.rounding`                                                                                                                                                                                                                                 | No import of `coola.iterator` in `src` (verified by `grep`); nothing to change.                                                                                                                                                           |
| `pyproject.toml` (`forbidden_modules` list)                                                                                                                                                                                                                      | `coola.iterator` stays in the list of the modules that the core must not import; no change.                                                                                                                                               |
| `tests/package_checks.py`, `tests/unit/test_public_api.py`, `tests/unit/test_default_registries.py`, `tests/unit/registry/test_dispatch.py`, `tests/integration/test_dataclasses.py`, `tests/properties/test_traversal.py`, `tests/properties/test_recursive.py` | Import the flat names (`from coola.iterator import ChildFinderRegistry, ...`); the test of the default registries has one registry to check instead of two.                                                                               |
| Third-party plugins of `coola.iterator.dfs` / `.bfs`                                                                                                                                                                                                             | Must register a `ChildFinder` in `coola.iterator.child_finders`. A DFS `BaseIterator` becomes a `find_children` that returns the children instead of recursing (the `ReversedListIterator` of the guide becomes `return reversed(data)`). |

## Decisions of the maintainers

All the decisions are taken. D1 to D8 and D10 follow the recommendation; D9 changes the plan: the
paths (`dfs_items`) are wanted, not now, and the design must make them easy to add.
The sections above are written for the decisions as taken.

**D1. One child finder and one registry for DFS and BFS?** _Recommended: yes._ The two packages
differ only by the order in which the same children are visited. One `find_children` per type,
walked by two engines, removes the duplicated classes (10 for 5 ideas), the duplicated registries,
singletons and plugin groups, the `bootstrap` module, and makes "same leaves" a property of the
design instead of a convention. The cost is a breaking rename and a custom DFS iterator that chose
its own order (`ReversedListIterator`) can no longer do it for DFS only: it returns its children in
the order they are visited, for both orders. Alternative: keep two packages and two registries;
cheaper for the release, but every future feature (a third order, a depth limit, the paths) is
written twice.

Decision: yes

**D2. Two functions or one with an `order` argument?** _Recommended: two functions._
`dfs_iterate(data)` and `bfs_iterate(data)` read well, are in the guide and in `recursive`'s
property ("the order of `dfs_iterate`"), and an `order="dfs"` string would add an option that needs
validation. Both call one function of `engine.py` each.

Decision: two functions

**D3. Which containers are entered by default?** _Recommended: the list of `recursive`_ (`list`,
`tuple`, `deque`, `Sequence`, `set`, `frozenset`, `dict`, `Mapping`, `MappingProxyType`, the
dataclasses), so that the same data gives the same leaves in `iterator`, `recursive`, `hashing` and
`summary` (constraint 4). The consequence is that `range(3)` becomes a leaf (it is a lazy arithmetic
sequence, like an array) and that a user class that only inherits `Collection` or `collections.abc.Set`
is a leaf until it registers a finder; the migration page says so. Alternative: widen `recursive` to
`Collection`; rejected, because `recursive` must rebuild the type and cannot for an arbitrary
`Collection`.

Decision: follow the recommendation

**D4. Name of the plugin group.** _Recommended: `coola.iterator.child_finders`_, the same form as
`coola.recursive.transformers`, `coola.equality.testers`, `coola.hashing.hashers`. The two old groups
are not read any more; the migration page lists the change. Alternative: keep reading the old
groups; rejected, because it keeps the two-registry design alive.

Decision: `coola.iterator.child_finders`

**D5. Keep `filter_by_type`?** _Recommended: keep, fix the type._ It is three lines, documented,
tested and used in the guide with `dfs_iterate`; removing it saves nothing and moves users to a
generator expression that the type checker types less well. Alternative: delete it and show
`(x for x in dfs_iterate(data) if isinstance(x, int))` in the guide; also defensible, and the only
cost is the migration note.

Decision: keep, fix the type

**D6. Public name `ChildFinder` or `Expander`.** _Recommended: `ChildFinder`/`BaseChildFinder`_ (the
names of the BFS package, which are the better ones: the method is `find_children`, the registry
method is `find_child_finder`). `recursive` uses `expand` because it returns the children and the
rebuild function; the two names differ because the two contracts differ.

Decision: `ChildFinder`/`BaseChildFinder`

**D7. Share the registry with `recursive`?** _Recommended: not now._ A `Transformer.expand` returns
the children and a rebuild function; the iterator needs only the children. One registry for both
would make `dfs_iterate` build a closure per container and couple two packages that have different
extension points (a custom type that can be traversed but not rebuilt is a normal case). Revisit if
a third traversal package (`summary`, `hashing`) wants the same children; the contract
`find_children` is a subset of `expand`, so the merge stays possible.

Decision: no

**D8. Memoize the shared containers?** _Recommended: no._ A container reachable by several paths
gives its leaves once per path (`[s, s]` gives them twice). This is the property "Leaves only,
once per occurrence", and `recursive_apply` calls `func` the same number of times. Cost: a DAG with
millions of paths is slow; that is documented, not hidden.

Decision: no

**D9. Yield the path of each leaf?** _Decided: yes, later; design for it now._ A `dfs_items(data)`
that yields `(path, leaf)` answers "where is the NaN" and is the natural next feature. It is not part
of this work, but nothing in this design may make it harder: lists the choices that keep
it cheap (the frame of the engine, the single place that knows the shape of a type, the path parts
shared with `equality`, the cached lookup) and the additive change it needs (an optional `find_items` on the
finder, a third engine loop). The hot path of `dfs_iterate` and `bfs_iterate` stays free of labels.
Same direction as D4 of `recursive.md`.

**D10. `registry.iterate` kept as a method?** _Recommended: removed._ `recursive` keeps
`TransformerRegistry.transform` as the registry-level entry point; here the registry is a dispatcher
and the functions `dfs_iterate` and `bfs_iterate` are the only entry points, so there is one way to
traverse and no method that has to choose an order.

Decision: removed

## Not covered

- Keys and paths: no iterator over the keys, the items or the paths of the leaves in this work.
  `dfs_items(data)` is decided and planned; says what the design keeps ready for it.
- Pydantic models, attrs classes and plain objects with `__dict__`: leaves, as before. `equality`
  enters pydantic models; a lazily registered `PydanticChildFinder` is a separate item, with the same
  one for `recursive`.
- NumPy arrays, torch tensors, data frames: leaves (the function receives the whole array).
  Element-wise iteration is the job of the caller.
- Parallel or async traversal, and a `max_depth` or `max_leaves` option: out of scope; `itertools.islice`
  stops a traversal after `n` leaves, because the functions are lazy.
- `coola.recursive`: the engine is separate (it must rebuild) and already iterative; only the
  container table and the order are shared.

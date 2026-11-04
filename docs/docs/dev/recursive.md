# Architecture of `coola.recursive`

:book: This page describes the design of the `coola.recursive` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/recursive.md#properties); the API is in the [reference](../refs/recursive.md).

## Overview

`recursive_apply` uses one engine (`coola.recursive.engine`) and one extension point. A transformer
only describes the shape of a type: `expand(data)` returns `None` for a leaf, or the children and
the function that rebuilds the container from the transformed children. The engine owns the
invariants: the transformer lookup (by MRO, cached for the call), the call of `func` on the leaves,
the cycle check and the depth (an explicit stack, so there is no `RecursionError`). This is the same
split as the equality engine, where a tester returns the child pairs.

## Constraints

1. **Simple and easy to maintain.** One function, one engine, one extension point. A transformer only
   describes a shape; there is no thread-local cycle state, no recursion-limit translation and no
   per-type workaround in the callers.
2. **Fast.** `recursive_apply` is the engine of `nested`, `jsonable` and `round_values`. The benchmark
   gate must hold; a design that is simpler but slower is rejected.
3. **Consistent with the siblings.** The same input is handled the same way by `equality`,
   `hashing`, `summary`, `iterator` and `recursive` (dataclasses are entered, iterators are leaves,
   cycles are detected, keyword-only options).
4. **Optional libraries stay lazy.** The package imports no optional library; NumPy arrays, pandas
   frames, ... are leaves unless a transformer is registered (the plugin group
   `coola.recursive.transformers` is the extension point).

## Architecture in detail

```
recursive_apply(data, func, *, registry=None)
        |   one registry (default: the singleton), one call = one engine run
        v
transform(data, func, registry)             coola/recursive/engine.py
        |   explicit stack of frames; per-call type -> transformer cache; ancestors' ids
        |
        |-- transformer = registry.find_transformer(type(data))        MRO, fallbacks
        |-- expansion   = transformer.expand(data)                     pure: no func, no recursion
        |        None                          -> leaf: func(data)
        |        (children, rebuild)           -> push the children; when they are done:
        |                                          rebuild(transformed_children)
        v
the transformed structure
```

The engine owns the invariants (dispatch cache, leaf call, cycles, depth, order). A transformer
**only describes the shape of a type**: its children and how to rebuild it. That is the same split
as `equality` (the engine owns identity, type, depth and cycles; a tester returns the child pairs).
`func` and the registry are not arguments of the transformer, so a transformer cannot forget a
protection and cannot recurse.

### The transformer contract

```python
Expansion = tuple[Sequence[Any], Callable[[list[Any]], Any]]


class Transformer(Protocol[T_contra]):
    def expand(self, data: T_contra) -> Expansion | None:
        """``None`` for a leaf; otherwise the children and the function that rebuilds ``data``
        from the transformed children (a list of the same length, in the same order)."""


class BaseTransformer(ABC, Generic[T]):  # convenience base class
    @abstractmethod
    def expand(self, data: T) -> Expansion | None: ...
```

Rules of the contract (documented in `BaseTransformer`, checked by one parametrised unit test over
the default registry, like `NATIVE_SAMPLES` in `equality`):

1. `expand` is pure and cheap: it does not call `func`, does not call the registry and does not
   recurse. It does not modify `data`.
2. `rebuild` receives a new `list` of the transformed children, in the order of `children`, and
   returns a **new** container (it never returns `data` itself unless `data` has no children).
3. An empty `children` is a container with no leaf (`[]`, `{}`): `rebuild([])` builds an empty
   container of the same type.
4. A leaf returns `None`: the engine calls `func(data)`.

### The engine

```python
def transform(data, func, registry):
    cache: dict[type, Transformer] = {}  # type -> transformer, for this call only
    ancestors: set[int] = set()  # ids of the containers being transformed

    def enter(obj):  # -> a frame, or None for a leaf
        t = type(obj)
        transformer = cache.get(t)
        if transformer is None:
            transformer = cache[t] = registry.find_transformer(t)
        expansion = transformer.expand(obj)
        if expansion is None:
            return None
        if id(obj) in ancestors:
            raise ValueError(f"Cycle detected: an object of type {t} contains itself")
        ancestors.add(id(obj))
        children, rebuild = expansion
        return Frame(obj, rebuild, iter(children), [])  # results is the output list

    frame = enter(data)
    if frame is None:
        return func(data)
    stack = [frame]
    while True:
        top = stack[-1]
        for child in top.children:  # resumes where it stopped
            child_frame = enter(child)
            if child_frame is None:
                top.results.append(func(child))
            else:
                stack.append(child_frame)
                break
        else:  # children exhausted: rebuild and hand the result to the parent
            stack.pop()
            ancestors.discard(id(top.obj))
            result = top.rebuild(top.results)
            if not stack:
                return result
            stack[-1].results.append(result)
```

Design points:

- **No `RecursionError`, no thread-local, no lock.** The state (stack, cache, ancestors) is local to
  the call, so threads and re-entrant calls (a `func` that calls `recursive_apply`) are independent.
- **Cycle check after `expand`** so that a leaf (an iterator, a string) never enters the set, and a
  container is reported even if a custom transformer returns its own object as a child.
- **The per-call cache** removes the 371 ns lookup per node. Nothing changes if the registry is
  modified while a call runs: the call keeps the transformers it resolved.
- `func` is called inside the loop: an exception propagates with the user's traceback, with no frame
  of the package between.
- The children are consumed lazily through an iterator, so a transformer can return a generator.

### Built-in transformers

| Transformer            | `expand(data)`                                                                                                                              | Default registration                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `DefaultTransformer`   | `None` (leaf)                                                                                                                               | `object`, `str`, `UserString`, `int`, `float`, `complex`, `bool` |
| `IdentityTransformer`  | `((), lambda _: data)`: no leaf, `func` never called, the value is kept as is (use for a type that must be preserved, or to skip a subtree) | none                                                             |
| `SequenceTransformer`  | `(list(data), lambda xs: rebuild_sequence(data, xs))`                                                                                       | `list`, `tuple`, `deque`, `Sequence`                             |
| `SetTransformer`       | `(list(data), lambda xs: rebuild_set(data, xs))`                                                                                            | `set`, `frozenset`                                               |
| `MappingTransformer`   | `(list(data.values()), lambda xs: rebuild_mapping(data, dict(zip(data, xs))))`                                                              | `dict`, `Mapping`, `MappingProxyType`                            |
| `DataclassTransformer` | `([getattr(data, f.name) for f in fields(data)], lambda xs: clone with the fields set)`                                                     | fallback: `dataclasses.is_dataclass`                             |
| `PydanticModelTransformer` | `([getattr(data, n) for n in model_fields + extra], lambda xs: data.model_copy(update=...))`, lazy: `coola.recursive.pydantic`        | `pydantic.BaseModel`                                             |

The registry API is `TransformerRegistry` with `register`, `register_many`, `find_transformer`,
`has_transformer`, `get_default_registry` (and its alias `get_default_transformer_registry`) and
`register_transformers`. `Transformer` (a protocol) and `BaseTransformer` (an ABC) follow the family
convention (`Hasher`/`BaseHasher`, `Summarizer`/`BaseSummarizer`): the protocol means a transformer
from another library needs no import of `coola`. `recursive_apply` is `recursive_apply(data, func, *,
registry=None)`.

The shortest extension example has no `func` and no registry to remember:

```python
class PointTransformer(BaseTransformer[Point]):
    def expand(self, data):
        return [data.x, data.y], lambda xs: Point(*xs)


register_transformers({Point: PointTransformer()})
```

### Rebuild rules (`coola.recursive.rebuild`)

All the rebuild rules live in `rebuild.py`, a documented public module (it is imported by `nested`)
with `rebuild_sequence`, `rebuild_mapping` and `rebuild_set`. It is not re-exported from
`coola.recursive`: it is the helper a custom container transformer needs, and re-exporting three more
names would grow the top-level namespace for little gain.

- `rebuild_*` call the constructor of `type(data)`; if it raises a `TypeError`, they raise a
  `TypeError` that names the type and says "register a transformer for `<type>`", with the original
  error as the cause. The same wrapper reports the unhashable set elements with the original message.
- Special cases are the three that exist (named tuple, `deque` `maxlen`, `defaultdict` factory).
  `ChainMap` is documented as flattened (its parents are not leaves of the structure), and is not
  special-cased: a user who needs it registers a transformer.
- `rebuild_mapping` and `rebuild_sequence` accept any iterable, not only `dict`/`list`, so the engine
  does not allocate an intermediate container.

### Dataclasses

A dataclass is entered by default, like in `equality`, `hashing` and `summary`, so
`recursive_apply(Config(...), f)` does not call `f` on the whole object. The rule is a fallback
(`dataclasses.is_dataclass`), so a user who wants a dataclass as a leaf registers `DefaultTransformer()`
for its type (a transformer registered for the type has priority over the fallback). An opt-in switch
on a process-wide singleton would be the worst kind of switch for a library.

A dataclass is rebuilt as a shallow copy in which every field is set with `object.__setattr__`. It
supports frozen and slotted dataclasses, transforms the `init=False` fields, supports an `InitVar`
without a default, and never runs user code (`__init__`, `__post_init__`), so the traversal has no side
effect. The cost is that a `__post_init__` that derives a field from another is not re-run, so a
derived field can become inconsistent with the field it was derived from after `func`; a user who needs
that registers a transformer. `dataclasses.replace` would re-validate through `__post_init__`, with the
limits of `init=False` fields and `InitVar`.

## Public API

```python
from coola.recursive import (
    recursive_apply,  # (data, func, *, registry=None)
    TransformerRegistry,
    get_default_registry,
    get_default_transformer_registry,  # alias
    register_transformers,  # (mapping, *, exist_ok=False)
    BaseTransformer,
    Transformer,  # protocol, expand() only
    DefaultTransformer,
    IdentityTransformer,
    SequenceTransformer,
    SetTransformer,
    MappingTransformer,
    DataclassTransformer,
)
```

`exist_ok` of `register_transformers` is keyword-only. `TransformerRegistry.transform(data, func)` is
the registry-level entry point and calls the engine.

## Dependents

| Package                                 | Use                                                                                                                                                        |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nested.keys`                           | A private `_KeyTransformer` registered in a copy of the default registry, and `rebuild_mapping`. `recursive` itself never transforms keys.                 |
| `nested.flat`                           | A private transformer for the nodes built by `from_flat_dict`.                                                                                             |
| `jsonable`                              | A `TransformerRegistry` on this engine (`ConvertTransformer`, a `SetTransformer` that sorts, ...).                                                         |
| `rounding.round_values(recursive=True)` | A dataclass is entered: its fields are rounded and the instance is rebuilt.                                                                                |
| `iterator`                              | The order of the calls of `func` equals the order of `dfs_iterate` (a property test ties the two); the three traversal packages share the container table. |

## Design choices and trade-offs

**Shared containers are not memoized.** Memoizing by `id` would fix the exponential cost of a DAG and
keep the aliasing of the input, but a container would be transformed once while `func` is called once
per occurrence for its leaves today; an impure `func` (a counter, a random draw, a call that logs)
would change behaviour, and the output would alias containers that the caller may then mutate
independently. A DAG with millions of paths is rare, and `dfs_iterate` also yields the leaves once per
path.

**The path of a failure is not reported.** With `expand` returning only the children, the path (a key,
an index) is not known. An exception raised by `func` would be more useful with the path
(`exc.add_note("at ['a'][2]")`), but that needs the label of each child (a second element in
`children`, as the `equality` testers return) and costs a tuple per child on the hot path. The `expand`
contract can be extended by returning `(label, child)` pairs without changing the structure of the
engine loop.

**Protocol and ABC.** `Transformer` (protocol) and `BaseTransformer` (ABC) match `Hasher`/`BaseHasher`
and `Summarizer`/`BaseSummarizer`; dropping the ABC would be a change of the three packages.

**The container table is shared by convention, not by code.** The iterator and the transformers have
different contracts (`expand` returns the children and the rebuild function; the iterator needs the
children only), so they keep separate registries.

## Not covered

- Keys: no transformer for the keys of a mapping in `recursive`. `nested` has a private one
  (`_KeyTransformer`); a public `MapKeysTransformer(key_func)` is one `expand` away if a second package
  needs it.
- Pydantic models, attrs classes and plain objects with `__dict__`: leaves. `equality`
  enters pydantic models; a `PydanticTransformer` is registered lazily the way the equality tester
  is (`register_lazy`) and is a separate item.
- NumPy arrays, torch tensors, data frames: leaves (`func` receives the whole array). Element-wise
  application is the job of `func`.
- Parallel or lazy (generator) application: the output is built whole. This is stated in the guide
  instead of promising laziness.
- The entry-point plugin group is `coola.recursive.transformers`; a plugin implements `expand`.

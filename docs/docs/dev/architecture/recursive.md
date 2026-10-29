# Architecture of `coola.recursive`

:book: This page describes the design of the `coola.recursive` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../../uguide/recursive.md#properties); the API is in the [reference](../../refs/recursive.md).

## Overview

`recursive_apply` uses one engine (`coola.recursive.engine`) and one extension point. A transformer
only describes the shape of a type: `expand(data)` returns `None` for a leaf, or the children and
the function that rebuilds the container from the transformed children. The engine owns the
invariants: the transformer lookup (by MRO, cached for the call), the call of `func` on the leaves,
the cycle check and the depth (an explicit stack, so there is no `RecursionError`). This is the same
split as the equality engine, where a tester returns the child pairs.

## Constraints

1. **Simple and easy to maintain.** The package is 1,360 lines for one function. The target removes
   the code that exists only to work around the design (the two-function hack of
   `KeyFilterTransformer`, the thread-local cycle state, the recursion-limit translation, the
   `_sets_to_lists` workaround of `nested`) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The deprecated paths are removed (the positional `registry`, the
   legacy mode of `KeyFilterTransformer`). The release notes list the changes.
3. **No slower than before.** `recursive_apply` is the engine of `nested` and `round_values`. The
   benchmark gate must hold; a design that is simpler but slower is rejected.
4. **Consistent with the siblings.** The same input must be handled the same way by `equality`,
   `hashing`, `summary`, `iterator` and `recursive` (dataclasses are entered, iterators are leaves,
   cycles are detected, keyword-only options).
5. **Optional libraries stay lazy.** The package imports no optional library; NumPy arrays, pandas
   frames, ... are leaves unless a transformer is registered (the plugin group
   `coola.recursive.transformers` is the extension point).

## Architecture in detail

```
recursive_apply(data, func, *, registry=None)
        |   one registry (default: the singleton), one call = one engine run
        v
transform(data, func, registry)             coola/recursive/engine.py   (about 40 lines)
        |   explicit stack of frames; per-call type -> transformer cache; ancestors' ids
        |
        |-- transformer = registry.find_transformer(type(data))        MRO, DataclassMarker
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
`func` and the registry are no longer arguments of the transformer, so a transformer cannot forget
a protection and cannot recurse.

### The transformer contract

```python
Expansion = tuple[Sequence[Any], Callable[[list[Any]], Any]]


class Transformer(Protocol[T_contra]):
    def expand(self, data: T_contra) -> Expansion | None:
        """``None`` for a leaf; otherwise the children and the function that rebuilds ``data``
        from the transformed children (a list of the same length, in the same order)."""


class BaseTransformer(ABC, Generic[T]):  # convenience base class, as before
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
  `raise_recursion_limit_error` is not used by this package any more.
- **Cycle check after `expand`** so that a leaf (an iterator, a string) never enters the set, and a
  container is reported even if a custom transformer returns its own object as a child.
- **The per-call cache** removes the 371 ns lookup per node. Nothing changes if the registry
  is modified while a call runs: the call keeps the transformers it resolved.
- `func` is called inside the loop: an exception propagates with the user's traceback, with no
  frame of the package between (the recursion of the previous version adds three per level).
- The children are consumed lazily through an iterator, so a transformer can return a generator
  (`KeyFilterTransformer` does not need to build a list of kept values first).

### Built-in transformers

| Transformer                           | `expand(data)`                                                                                                                                                                                                                                          | Default registration                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `DefaultTransformer`                  | `None` (leaf)                                                                                                                                                                                                                                           | `object`, `str`, `UserString`, `int`, `float`, `complex`, `bool` |
| `IdentityTransformer`                 | `((), lambda _: data)`: no leaf, `func` never called, the value is kept as is (use for a type that must be preserved, or to skip a subtree)                                                                                                             | none                                                             |
| `SequenceTransformer`                 | `(list(data), lambda xs: rebuild_sequence(data, xs))`                                                                                                                                                                                                   | `list`, `tuple`, `deque`, `Sequence`                             |
| `SetTransformer`                      | `(list(data), lambda xs: rebuild_set(data, xs))`                                                                                                                                                                                                        | `set`, `frozenset`                                               |
| `MappingTransformer`                  | `(list(data.values()), lambda xs: rebuild_mapping(data, dict(zip(data, xs))))`                                                                                                                                                                          | `dict`, `Mapping`, `MappingProxyType`                            |
| `DataclassTransformer`                | `([getattr(data, f.name) for f in fields(data)], lambda xs: clone with the fields set)`                                                                                                                                                   | `DataclassMarker`                                  |
| `KeyFilterTransformer(key_predicate)` | `MappingTransformer` on the entries whose key does not satisfy the predicate. One mode only: the predicate is stored in the transformer, `func` is always the function of the values (the legacy mode, `transform_func` and `_value_func` are deleted). | none (`nested.remove_keys_if` registers it)                      |

Removed: `ConditionalTransformer`, `enter_dataclasses`, the positional
`registry` of `recursive_apply` and the legacy mode of `KeyFilterTransformer`. Kept as they
are: `Transformer` protocol and `BaseTransformer` (the family convention: `Hasher`/`BaseHasher`,
`Summarizer`/`BaseSummarizer`), `TransformerRegistry` with `register`, `register_many`,
`find_transformer`, `has_transformer`, `get_default_registry` and its alias
`get_default_transformer_registry`, `register_transformers`. `recursive_apply` is
`recursive_apply(data, func, *, registry=None)`.

The extension example of the guide becomes the shortest possible one, with no `func` and no registry
to remember:

```python
class PointTransformer(BaseTransformer[Point]):
    def expand(self, data):
        return [data.x, data.y], lambda xs: Point(*xs)


register_transformers({Point: PointTransformer()})
```

### Rebuild rules (`coola.recursive.rebuild`)

All the rebuild rules live in `rebuild.py`, which becomes a documented public module (it is imported
by `nested`, ) with `rebuild_sequence`, `rebuild_mapping` and the new `rebuild_set`.

- `rebuild_*` call the constructor of `type(data)`; if it raises a `TypeError`, they raise a
  `TypeError` that names the type and says "register a transformer for `<type>`", with the original
  error as the cause. The same wrapper reports the unhashable set elements with the original
  message.
- Special cases stay the three that exist (named tuple, `deque` `maxlen`, `defaultdict` factory).
  `ChainMap` is documented as flattened (its parents are not leaves of the structure), and is not
  special-cased: a user who needs it registers a transformer.
- `rebuild_mapping` and `rebuild_sequence` accept any iterable, not only `dict`/`list`, so the
  engine does not allocate an intermediate container (keep the signatures simple: `items`).

## Public API

```python
from coola.recursive import (
    recursive_apply,  # (data, func, *, registry=None)
    TransformerRegistry,
    get_default_registry,
    get_default_transformer_registry,  # alias, unchanged
    register_transformers,  # (mapping, *, exist_ok=False)
    BaseTransformer,
    Transformer,  # protocol, expand() only
    DefaultTransformer,
    IdentityTransformer,
    SequenceTransformer,
    SetTransformer,
    MappingTransformer,
    DataclassTransformer,
    KeyFilterTransformer,
)
```

`register_transformers(mapping, exist_ok=False)` becomes keyword-only for `exist_ok` (the
`CLAUDE.md` rule: new options are keyword-only). `TransformerRegistry.transform(data, func)` stays
as the registry-level entry point and calls the engine, so that `registry.transform` in the guide
and in the doctests keeps working.

## Dependents

| Package                                 | Change                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nested.mapping.remove_keys_if`         | No change of behaviour. It keeps building its registry (`IdentityTransformer` for `object`, `KeyFilterTransformer(key_predicate=...)` for mappings). Import `rebuild_sequence` from `coola.recursive` (it becomes public) if it is still needed.                                                                                                       |
| `nested.conversion.convert_to_jsonable` | Passes a registry copy of the default one with `DataclassMarker: DefaultTransformer()` (a dataclass is converted whole by `to_jsonable`, as before; with D1 it would otherwise be entered and stay a dataclass) and a `SetToListTransformer` (`expand` returns the elements and `rebuild` returns `try_sorted_list`). `_sets_to_lists` is deleted. |
| `rounding.round_values(recursive=True)` | A dataclass is entered by default after D1: its fields are rounded and the instance is rebuilt (before: the dataclass was a leaf and `_round_leaf` returned it unchanged). A behaviour change to document.                                                                                                                                             |
| `iterator.dfs` / `iterator.bfs`         | Outside this design, but D1 must be applied to the three together: `enter_dataclasses` exists in all three and is opt-in in all three. Property: the order of the calls of `func` equals the order of `dfs_iterate`.                                                                                                                       |

## Decisions for the maintainers

**D1. Enter dataclasses by default?** _Recommended: yes._ `equality`, `hashing` and `summary`
enter them, so `recursive_apply(Config(...), f)` calling `f` on the whole object is the surprise
(25.3 of `suggestions.md`). The opt-in `enter_dataclasses()` mutates a process-wide singleton, which is
the worst kind of switch for a library. Consequences: `convert_to_jsonable` needs its own leaf
registry, `round_values(recursive=True)` rounds the fields of a dataclass, and a user
who wants a dataclass as a leaf registers `DefaultTransformer()` for its type (a transformer
registered for the type has priority over the marker, which already works). Alternative: keep it
opt-in, and keep `enter_dataclasses` with a `registry` argument; cheaper, but the three traversal
packages stay inconsistent with `equality`.

**D2. How a dataclass is rebuilt.** _Recommended: shallow copy and set every field with
`object.__setattr__`._ It supports frozen and slotted dataclasses, transforms the `init=False`
fields (before: silently skipped), supports an `InitVar` without a default (before: a `ValueError`), and
never runs user code (`__init__`, `__post_init__`), so the traversal has no side effect. Cost: a
`__post_init__` that derives a field from another is not re-run, so a derived field can become
inconsistent with the field it was derived from after `func`; the user who needs that registers a
transformer. Alternative: keep `dataclasses.replace` (re-validates through `__post_init__`, keeps
the previous limits).

**D3. Memoize the shared containers?** _Recommended: no._ Memoizing by `id` would fix the
exponential cost of a DAG and keep the aliasing of the input, but a container would be transformed
once while `func` is called once per occurrence for its leaves today; an impure `func` (a counter, a
random draw, a call that logs) changes behaviour, and the output would alias containers that the
caller may then mutate independently. A DAG with millions of paths is rare, and
`dfs_iterate` also yields the leaves once per path (the property "Every leaf is returned").

**D4. Should the engine report the path of the failure?** _Recommended: not now._ With `expand`
returning only the children, the path (a key, an index) is not known. An exception raised by
`func` would be more useful with the path (`exc.add_note("at ['a'][2]")`, Python 3.11 has
`add_note`), but that needs the label of each child (a second element in `children`, as the
`equality` testers return) and costs a tuple per child on the hot path. Revisit if users ask; the
`expand` contract can be extended by returning `(label, child)` pairs without touching the engine
loop's structure.

**D5. One protocol or two?** _Recommended: keep `Transformer` (protocol) and `BaseTransformer`
(ABC)._ Same as `Hasher`/`BaseHasher` and `Summarizer`/`BaseSummarizer`; the cost is a few lines and
the benefit is that a transformer from another library needs no import of `coola`. Dropping the
ABC would be a change of the three packages, not of this one.

**D6. Public `rebuild` module.** _Recommended: public (documented, in the guide), not re-exported
from `coola.recursive`._ It is the helper a custom container transformer needs. Re-exporting three
more names grows the top-level namespace for little gain; the guide names the module.

## Not covered

- Keys: no transformer for the keys of a mapping (`add_prefix_suffix_to_keys` in `nested` does its
  own traversal). A `MapKeysTransformer(key_func)` is the natural next transformer once the engine
  exists; it is one `expand` and is not part of this design.
- Pydantic models, attrs classes and plain objects with `__dict__`: leaves, as before. `equality`
  enters pydantic models; a `PydanticTransformer` is registered lazily the way the equality tester
  is (`register_lazy`) and is a separate item.
- NumPy arrays, torch tensors, data frames: leaves (`func` receives the whole array). Element-wise
  application is the job of `func`.
- Parallel or lazy (generator) application: the output is built whole. This is stated in the guide
  instead of promising laziness.
- `iterator.dfs` explicit stack: same technique, separate design.
- The entry-point plugin group `coola.recursive.transformers` keeps its name; a plugin that
  implements `transform(data, func, registry)` must be ported to `expand` (migration page).

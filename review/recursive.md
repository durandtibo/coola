# `coola.recursive`: target state and the changes to reach it

This document describes the design that `coola.recursive` should have, then the changes to make to
the current implementation (`main`, `698a6808`) to get there. The design follows the rules of the
user guide (`docs/docs/uguide/recursive.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: registry dispatch, MRO lookup, lazy imports, strict contracts), and
it stays aligned with the sibling packages `equality`, `hashing`, `summary` and `iterator`. Its aim is
a small package that is easy to maintain: **one engine, one extension point, one rebuild rule per
container type**. Breaking changes are accepted, and the future state is more important than the
past one.

Evidence comes from probes and a throw-away prototype run on `main` (Python 3.14, scripts kept
outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from the
code. Nothing in `src`, `tests` or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 1,360 lines for one function. The target removes
   the code that exists only to work around the design (the two-function hack of
   `KeyFilterTransformer`, the thread-local cycle state, the recursion-limit translation, the
   `_sets_to_lists` workaround of `nested`) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The deprecated paths are removed (the positional `registry`, the
   legacy mode of `KeyFilterTransformer`). The release notes list the changes.
3. **No slower than today.** `recursive_apply` is the engine of `nested` and `round_values`. The
   benchmark gate of section 8 must hold; a design that is simpler but slower is rejected.
4. **Consistent with the siblings.** The same input must be handled the same way by `equality`,
   `hashing`, `summary`, `iterator` and `recursive` (dataclasses are entered, iterators are leaves,
   cycles are detected, keyword-only options).
5. **Optional libraries stay lazy.** The package imports no optional library; NumPy arrays, pandas
   frames, ... are leaves unless a transformer is registered (the plugin group
   `coola.recursive.transformers` is the extension point).

## 1. Properties of the sub-package

These are the rules of the user guide. Every design choice below exists to make one of them true
**by construction** (in the engine) instead of by the discipline of each transformer.

| Property                 | Statement                                                                                                                                                                                                    | Enforced by                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- |
| Leaves only              | `func` is called on the leaves, exactly once per occurrence, never on a container. A leaf is a value whose transformer has no children (scalars, `str`, `bytes`, iterators, arrays, ...).                    | the engine                               |
| Structure-preserving     | A container is rebuilt from the transformed children with its original type (a subclass, a named tuple, a `defaultdict` factory and a `deque` `maxlen` are kept). Keys are never transformed.                | one rebuild rule per container type      |
| Type dispatch            | The transformer is chosen from `type(data)` along the Method Resolution Order, so the most specific registered type wins. A transformer is always found: `object` is registered by the default registry.     | the registry                             |
| Registry-extensible      | A new type is supported by registering one transformer, without changing another one and without subclassing a base class (a protocol is enough).                                                            | the registry, the `Transformer` protocol |
| Deterministic order      | The leaves are visited depth first, left to right: the order of `dfs_iterate`. The elements of a set are visited in the iteration order of the set (not specified).                                          | the engine                               |
| Pure                     | The input is never modified. The output shares no container with the input (leaves can be shared). Transformers keep no state: a registry and a transformer can be shared by threads and calls.              | the engine, stateless transformers       |
| Cycle-safe               | A container that contains itself raises `ValueError`, whichever transformer is registered for it.                                                                                                            | the engine                               |
| Any depth                | No `RecursionError`, whatever the nesting or the caller's stack: the engine uses an explicit stack. The depth is bounded by memory only.                                                                     | the engine                               |
| Errors propagate         | An exception of `func` propagates unchanged. A container that cannot be rebuilt raises a `TypeError` that names the type and tells how to register a transformer. Unhashable set elements raise `TypeError`. | the engine, the rebuild rules            |
| Consistent with siblings | Dataclasses are entered (like `objects_are_equal`, `hash_object`, `summarize`); iterators and generators are leaves (like `dfs_iterate`); options are keyword-only.                                          | the default registry                     |
| Fast                     | The cost is linear in the number of nodes; the transformer lookup is cached per call; no lock, no thread-local, no per-node bookkeeping outside the cycle check.                                             | the engine                               |

Not guaranteed, and documented as such:

- **Shared references.** A container reachable by several paths (a DAG) is transformed once per
  path, so `func` is called once per occurrence, the output has no aliasing, and the cost is
  proportional to the number of paths, not to the number of objects (a list nested 22 times with
  `[x, x]` has 2^22 leaves and takes 10 s today **[verified]**). No memoization: it would change
  how often an impure `func` is called (decision D3).
- **Order of the calls inside a set** and the result of `func` that is not deterministic.
- **Subclasses whose constructor differs** from the base type (see the rebuild rules of section 4.4).

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Status         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The protections live in the registry, but the recursion lives in each transformer.** Every transformer calls `registry.transform` for its children; the cycle check, the thread-local `active` set and the `RecursionError` translation in `TransformerRegistry.transform` work only if the transformer cooperates. A custom transformer that recurses itself bypasses all of them.                                                                                                                                                                                                    | **[reading]**  |
| F2  | **Depth.** The recursion costs about three frames per level, and a 3,000-level list raises a `RecursionError` (with a good message). The explicit stack of the prototype transforms 50,000 levels. `dfs_iterate` has the same limit; `bfs_iterate` and `objects_are_equal` have none.                                                                                                                                                                                                                                                                                                    | **[verified]** |
| F3  | **Two functions through one slot.** `transform(data, func, registry)` threads one function through the recursion, so a transformer that needs a predicate for the keys and a function for the values cannot be written. `KeyFilterTransformer` works around it with a deprecated "legacy mode" (`func` is the predicate, `transform_func`, `_value_func`, an `isinstance` check for containers): 60 lines.                                                                                                                                                                               | **[reading]**  |
| F4  | **`ConditionalTransformer` is dead weight.** Nothing in `src` uses it. `ConditionalTransformer(DefaultTransformer(), cond)` is `lambda x: f(x) if cond(x) else x`, a guard in `func`; for a container it skips the subtree, which `IdentityTransformer` registered for the type already does.                                                                                                                                                                                                                                                                                            | **[verified]** |
| F5  | **Dataclasses are leaves by default**, the opposite of `equality`, `hashing` and `summary`, and entering them needs `enter_dataclasses()`, which mutates the process-wide default registry. `DataclassTransformer` uses `dataclasses.replace`: a field with `init=False` is silently **not** transformed (it gets the value computed by `__init__`), and an `InitVar` without a default raises.                                                                                                                                                                                          | **[verified]** |
| F6  | **Rebuild.** `type(data)(items)` raises a raw constructor `TypeError` for a subclass such as `class Sub(list): def __init__(self, x, extra)` (`Sub.__init__() missing 1 required positional argument`). A `ChainMap` is rebuilt as a `ChainMap` with a single flat map: the parents are lost. `SetTransformer` rebuilds on its own instead of in `rebuild.py`.                                                                                                                                                                                                                           | **[verified]** |
| F7  | **`nested.conversion._sets_to_lists`** exists because a transformer cannot change the type of its container: the sets are converted by a second pass over the result.                                                                                                                                                                                                                                                                                                                                                                                                                    | **[reading]**  |
| F8  | **Deprecated paths kept.** `recursive_apply(data, func, registry)` (positional) and the legacy mode of `KeyFilterTransformer` still emit `DeprecationWarning`.                                                                                                                                                                                                                                                                                                                                                                                                                           | **[verified]** |
| F9  | **`ARG002` noqa.** Three `# noqa: ARG002` in the package (`DefaultTransformer`, `IdentityTransformer`) exist because the signature passes `func` and `registry` to transformers that need neither (`review/noqa-review.md`).                                                                                                                                                                                                                                                                                                                                                             | **[reading]**  |
| F10 | **The user guide contradicts the code.** It states "Memory-efficient generator-based traversal" and a property "Lazy traversal: the traversal relies on generators" (the code builds lists and the whole output; nothing is lazy), "Not guaranteed: self-referencing containers are not detected" (they raise `ValueError` since 25.1), and "MappingTransformer: recursive transformation of keys and values" (keys are never transformed). The `BaseTransformer.transform` example runs a `DefaultTransformer` on a list and shows `[1, 2, 3, 1, 2, 3]`, which teaches the wrong thing. | **[verified]** |
| F11 | **Lookup cost.** `find_transformer` costs 371 ns per node (a `dict.get` costs 48 ns) and is 70 % of the time of `recursive_apply` on a flat list of 100,000 ints (54 ms **[verified]**). Nothing is cached per call.                                                                                                                                                                                                                                                                                                                                                                     | **[verified]** |
| F12 | **Cross-package workaround of the public contract.** `nested.mapping` imports `coola.recursive.rebuild.rebuild_sequence`, a module that is not in `__all__`, and builds a registry of four transformer classes per call to remove keys.                                                                                                                                                                                                                                                                                                                                                  | **[reading]**  |

What is already right and must not regress: type dispatch with MRO and a cache; `UserString` and
`str` as leaves; iterators and generators as leaves; the rebuild of named tuples, `deque` (`maxlen`)
and `defaultdict` (factory); `OrderedDict`, `Counter` and `MappingProxyType` keep their type; keys
untouched; the input unmodified; keyword-only `registry`.

## 3. Architecture

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

### 3.1 The transformer contract

```python
Expansion = tuple[Sequence[Any], Callable[[list[Any]], Any]]


class Transformer(Protocol[T_contra]):
    def expand(self, data: T_contra) -> Expansion | None:
        """``None`` for a leaf; otherwise the children and the function that rebuilds ``data``
        from the transformed children (a list of the same length, in the same order)."""


class BaseTransformer(ABC, Generic[T]):  # convenience base class, as today
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

### 3.2 The engine

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
- **The per-call cache** removes the 371 ns lookup per node (F11). Nothing changes if the registry
  is modified while a call runs: the call keeps the transformers it resolved.
- `func` is called inside the loop: an exception propagates with the user's traceback, with no
  frame of the package between (the recursion of today adds three per level).
- The children are consumed lazily through an iterator, so a transformer can return a generator
  (`KeyFilterTransformer` does not need to build a list of kept values first).

### 3.3 Built-in transformers

| Transformer                           | `expand(data)`                                                                                                                                                                                                                                          | Default registration                                             |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `DefaultTransformer`                  | `None` (leaf)                                                                                                                                                                                                                                           | `object`, `str`, `UserString`, `int`, `float`, `complex`, `bool` |
| `IdentityTransformer`                 | `((), lambda _: data)`: no leaf, `func` never called, the value is kept as is (use for a type that must be preserved, or to skip a subtree)                                                                                                             | none                                                             |
| `SequenceTransformer`                 | `(list(data), lambda xs: rebuild_sequence(data, xs))`                                                                                                                                                                                                   | `list`, `tuple`, `deque`, `Sequence`                             |
| `SetTransformer`                      | `(list(data), lambda xs: rebuild_set(data, xs))`                                                                                                                                                                                                        | `set`, `frozenset`                                               |
| `MappingTransformer`                  | `(list(data.values()), lambda xs: rebuild_mapping(data, dict(zip(data, xs))))`                                                                                                                                                                          | `dict`, `Mapping`, `MappingProxyType`                            |
| `DataclassTransformer`                | `([getattr(data, f.name) for f in fields(data)], lambda xs: clone with the fields set)` (decision D2)                                                                                                                                                   | `DataclassMarker` (decision D1)                                  |
| `KeyFilterTransformer(key_predicate)` | `MappingTransformer` on the entries whose key does not satisfy the predicate. One mode only: the predicate is stored in the transformer, `func` is always the function of the values (the legacy mode, `transform_func` and `_value_func` are deleted). | none (`nested.remove_keys_if` registers it)                      |

Removed: `ConditionalTransformer` (F4), `enter_dataclasses` (decision D1), the positional
`registry` of `recursive_apply` and the legacy mode of `KeyFilterTransformer` (F8). Kept as they
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

### 3.4 Rebuild rules (`coola.recursive.rebuild`)

All the rebuild rules live in `rebuild.py`, which becomes a documented public module (it is imported
by `nested`, F12) with `rebuild_sequence`, `rebuild_mapping` and the new `rebuild_set`.

- `rebuild_*` call the constructor of `type(data)`; if it raises a `TypeError`, they raise a
  `TypeError` that names the type and says "register a transformer for `<type>`", with the original
  error as the cause (F6). The same wrapper reports the unhashable set elements with the original
  message.
- Special cases stay the three that exist (named tuple, `deque` `maxlen`, `defaultdict` factory).
  `ChainMap` is documented as flattened (its parents are not leaves of the structure), and is not
  special-cased: a user who needs it registers a transformer.
- `rebuild_mapping` and `rebuild_sequence` accept any iterable, not only `dict`/`list`, so the
  engine does not allocate an intermediate container (keep the signatures simple: `items`).

## 4. Public API after the change

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

## 5. Dependents

| Package                                 | Change                                                                                                                                                                                                                                                                                                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nested.mapping.remove_keys_if`         | No change of behaviour. It keeps building its registry (`IdentityTransformer` for `object`, `KeyFilterTransformer(key_predicate=...)` for mappings). Import `rebuild_sequence` from `coola.recursive` (it becomes public) if it is still needed.                                                                                                       |
| `nested.conversion.convert_to_jsonable` | Passes a registry copy of the default one with `DataclassMarker: DefaultTransformer()` (a dataclass is converted whole by `to_jsonable`, as today; with D1 it would otherwise be entered and stay a dataclass) and a `SetToListTransformer` (`expand` returns the elements and `rebuild` returns `try_sorted_list`). `_sets_to_lists` is deleted (F7). |
| `rounding.round_values(recursive=True)` | A dataclass is entered by default after D1: its fields are rounded and the instance is rebuilt (before: the dataclass was a leaf and `_round_leaf` returned it unchanged). A behaviour change to document.                                                                                                                                             |
| `iterator.dfs` / `iterator.bfs`         | Outside this design, but D1 must be applied to the three together: `enter_dataclasses` exists in all three and is opt-in in all three. Property: the order of the calls of `func` equals the order of `dfs_iterate` (section 8).                                                                                                                       |

## 6. Evidence: a prototype of the engine

A prototype of the loop above (explicit stack, the real `rebuild_*` helpers, the real default
registry for the lookup with a per-call cache, `expand` inlined) was run on `main`. It is not part of
the repository. **[verified]**

| Input                                                                   | `main`                            | Prototype |
| ----------------------------------------------------------------------- | --------------------------------- | --------- |
| tree: depth 8 of dict / list / tuple / set / floats (about 1,500 nodes) | 1.8 ms                            | 1.5 ms    |
| flat list of 100,000 ints                                               | 54.1 ms                           | 26.3 ms   |
| list nested 50,000 levels deep                                          | `RecursionError` (at about 3,000) | works     |
| output equals the output of `main` on the tree                          | n/a                               | yes       |

Without the per-call cache (a registry lookup per node) the flat list costs 54 ms, like `main`:
the cache is what makes the engine faster, so it is part of the design and not an optimisation to
add later. The real `expand` is a method call and builds a tuple per container, so the real engine
will be somewhat slower than the prototype; the gate (section 8) is "no slower than `main`" on both
inputs, with the margin above.

The dataclass clone of D2 was checked on a frozen, slotted dataclass with an `init=False` field and
an `InitVar`: `copy.copy` plus `object.__setattr__` for each field gives
`C(a='x', b='y')`; `dataclasses.replace` ignores the `init=False` field.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                       | Breaking                  |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| 1    | **Fix the guide (F10).** Remove "generator-based / memory-efficient / lazy", remove "self-referencing containers are not detected" (replace by the cycle and depth properties), fix "keys and values", fix the `BaseTransformer.transform` docstring example, add the **Properties** table of section 1 at the top of the page and in the package docstring. Documentation only.                             | no                        |
| 2    | **Delete the dead and deprecated code (F4, F8).** `ConditionalTransformer`, the positional `registry`, the legacy mode of `KeyFilterTransformer` (with `transform_func`, `_value_func`, `_identity`). Update `__all__`, the guide, the tests and the status table. `register_transformers(..., *, exist_ok=False)`.                                                                                          | yes                       |
| 3    | **Rebuild module (F6, F12).** Move the set rebuild to `rebuild.py` (`rebuild_set`), make the module public, wrap the constructor `TypeError` with the type name and the remedy, document `ChainMap`.                                                                                                                                                                                                         | no (messages)             |
| 4    | **Engine and `expand` protocol (F1, F2, F9, F11).** Add `engine.py`, change `Transformer`/`BaseTransformer` to `expand`, port the seven built-in transformers, `TransformerRegistry.transform` delegates to the engine, delete the thread-local state and the recursion-limit translation, delete the `ARG002` noqa. Rewrite the unit tests of each transformer on `expand`. Benchmark gate.                 | yes (custom transformers) |
| 5    | **Dataclasses by default (F5, D1, D2).** Register `DataclassTransformer` on `DataclassMarker` in the default registry, clone with the fields set, delete `enter_dataclasses` and `tests/integration/test_enter_dataclasses.py`'s opt-in test (keep the behaviour tests). `convert_to_jsonable` gets its leaf registry. Apply the same decision to `iterator.dfs` and `iterator.bfs` in their own PR.         | yes                       |
| 6    | **`convert_to_jsonable` without the second pass (F7).** `SetToListTransformer`, delete `_sets_to_lists`. Can be merged into step 5.                                                                                                                                                                                                                                                                          | no                        |
| 7    | **Documentation.** Rewrite `uguide/recursive.md` (extension example, properties, DAG and set-order notes), the `recursive` paragraph of `dev/architecture.md`, the migration page (`migration/v0.11_to_v1.0.md`: the `expand` protocol, `enter_dataclasses`, `ConditionalTransformer`, positional `registry`), and the status table of `review/suggestions.md` (items 25.1, 25.3, 28.2 with the PR numbers). | no                        |

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/recursive/`); the conventions of `CLAUDE.md` apply
(one behaviour per test, literal expected values, a regression test for every bug, no local imports).

- **Engine** (`test_engine.py`, new):
  - the order of the calls of `func` on a nested list/dict/tuple/set structure (a list that records
    the leaves, the whole recorded list asserted against a literal);
  - `func` is called once per occurrence for a shared container (`[s, s]`);
  - a cycle through a list, a dict and a tuple-in-list raises `ValueError`, whichever transformer
    is registered for the type (a custom transformer that returns its own object as a child);
  - depth 100,000 does not raise (regression for F2; check it fails on `main` with
    `git stash push src`);
  - an exception of `func` propagates unchanged and leaves no state: the next call works;
  - re-entrant call (`func` calls `recursive_apply`) and a custom registry are independent;
  - the per-call cache: a transformer registered during a call is not used by this call;
  - `TypeNotRegisteredError` for an empty registry (as today).
- **Each transformer** (`test_<module>.py`): `expand` asserts the whole output against a literal
  (children list and the rebuilt value), and the typical, empty, single-element and duplicate
  cases; subclasses keep their type; named tuple, `deque(maxlen)`, `defaultdict`, `OrderedDict`,
  `Counter`, `MappingProxyType`; the input is not modified and the output shares no container.
  `KeyFilterTransformer`: predicate at every depth, in sequences and in sets, `func` applied to the
  values, an empty result, and the predicate called once per key.
- **Rebuild** (`test_rebuild.py`): the error that names the type for `Sub(list)`, the original
  error as the cause, `ChainMap` documented behaviour.
- **Contract test over the default registry**: every transformer of the default registry returns
  `None` or a `(children, rebuild)` whose rebuild of the unchanged children is equal to the sample
  (one sample per registered type, like `NATIVE_SAMPLES`).
- **Properties** (`tests/properties/test_traversal.py`, extend): identity returns equal data (kept),
  the type of the containers is kept (kept), **the order of the calls of `func` equals the order of
  `dfs_iterate`** (new, ties `recursive` and `iterator`), the input is not modified (new).
- **Integration** (`tests/integration/`): threads running `recursive_apply` on the default registry
  with a lock around the shared list (`run_threads`, docstring "Test that ..."); the `nested` and
  `round_values` end-to-end behaviour with the real registry, with no patching; the dataclass
  default with the default registry (frozen, slots, `init=False`, `InitVar`, nested dataclass,
  a dataclass registered with its own transformer has priority).
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Benchmark gate** (`benchmarks/`, the CI benchmark): `recursive_apply` on the tree and on the flat
  list of section 6; the new engine must not be slower than `main` on either (target: at most the
  numbers of `main`; expected: faster). A failing gate rejects the step.
- Coverage: 100 % of `src/coola/recursive` (`--cov=coola`), no `# pragma: no cover`.

## 9. Decisions for the maintainers

**D1. Enter dataclasses by default?** _Recommended: yes._ `equality`, `hashing` and `summary`
enter them, so `recursive_apply(Config(...), f)` calling `f` on the whole object is the surprise
(25.3 of `suggestions.md`). The opt-in `enter_dataclasses()` mutates a process-wide singleton, which is
the worst kind of switch for a library. Consequences: `convert_to_jsonable` needs its own leaf
registry (section 5), `round_values(recursive=True)` rounds the fields of a dataclass, and a user
who wants a dataclass as a leaf registers `DefaultTransformer()` for its type (a transformer
registered for the type has priority over the marker, which already works). Alternative: keep it
opt-in, and keep `enter_dataclasses` with a `registry` argument; cheaper, but the three traversal
packages stay inconsistent with `equality`.

**D2. How a dataclass is rebuilt.** _Recommended: shallow copy and set every field with
`object.__setattr__`._ It supports frozen and slotted dataclasses, transforms the `init=False`
fields (today silently skipped), supports an `InitVar` without a default (today a `ValueError`), and
never runs user code (`__init__`, `__post_init__`), so the traversal has no side effect. Cost: a
`__post_init__` that derives a field from another is not re-run, so a derived field can become
inconsistent with the field it was derived from after `func`; the user who needs that registers a
transformer. Alternative: keep `dataclasses.replace` (re-validates through `__post_init__`, keeps
today's limits).

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

## 10. Not covered

- Keys: no transformer for the keys of a mapping (`add_prefix_suffix_to_keys` in `nested` does its
  own traversal). A `MapKeysTransformer(key_func)` is the natural next transformer once the engine
  exists; it is one `expand` and is not part of this design.
- Pydantic models, attrs classes and plain objects with `__dict__`: leaves, as today. `equality`
  enters pydantic models; a `PydanticTransformer` is registered lazily the way the equality tester
  is (`register_lazy`) and is a separate item.
- NumPy arrays, torch tensors, data frames: leaves (`func` receives the whole array). Element-wise
  application is the job of `func`.
- Parallel or lazy (generator) application: the output is built whole. This is stated in the guide
  instead of promising laziness (F10).
- `iterator.dfs` explicit stack (25.1 of `suggestions.md`): same technique, separate design.
- The entry-point plugin group `coola.recursive.transformers` keeps its name; a plugin that
  implements `transform(data, func, registry)` must be ported to `expand` (migration page).

## Appendix: effect on the size of the package

| Module              | Lines today | Expected after | Why                                                                            |
| ------------------- | ----------- | -------------- | ------------------------------------------------------------------------------ |
| `key.py`            | 172         | about 70       | one mode, no `transform_func`, no `_value_func`, no deprecation warning        |
| `conditional.py`    | 88          | 0              | removed                                                                        |
| `registry.py`       | 219         | about 150      | no thread-local, no recursion translation; `transform` is a call to the engine |
| `interface.py`      | 254         | about 190      | no `enter_dataclasses`, no positional shim                                     |
| `base.py`           | 99          | about 90       | `expand` replaces `transform` (docstrings are longer than the signature)       |
| `engine.py`         | 0           | about 70       | new                                                                            |
| `rebuild.py`        | 74          | about 110      | `rebuild_set`, the error that names the type                                   |
| other transformers  | 400         | about 330      | no `func`/`registry` parameters, no `ARG002` noqa                              |
| `nested.conversion` | n/a         | minus 15       | `_sets_to_lists` deleted                                                       |

The line counts are estimates from the prototype and from the code to delete; the gate is the
behaviour (sections 1 and 8), not the size.

## Appendix: benchmark of the old and the new implementation

Comparison of `tests/benchmarks/test_recursive_benchmark.py` on the implementation before the change
and on the new engine. The benchmarks only use
`recursive_apply(data, func)` with the default registry, so they run unchanged on the new design.
Compare with `inv test.benchmark` (or `pytest tests/benchmarks/test_recursive_benchmark.py
--benchmark-only --benchmark-save=<name>`, then `--benchmark-compare=<name>`).

Machine: Apple M1, macOS (Darwin), Python 3.14.8, `pytest-benchmark`. Both implementations were run
back to back on the same machine (old: `273015ac`, `src/coola/recursive` identical to `b3b8753f`,
checked out in a throw-away worktree with the current benchmark file; new: `main` at `d097dbda`,
with the engine and the `expand` protocol). One run each, nothing else heavy running. `func` is
`x * 2`, except identity for `flat_list_identity`, `flat_list_of_strings`, `list_of_small_dicts`
and `mixed_containers`. The speed-up is old time / new time: above 1 is faster (5.00x is five times
faster), below 1 is slower. The absolute numbers depend on the machine; read the speed-ups.

The raw results are saved in `.benchmarks/Darwin-CPython-3.14-64bit/`:
`0002_recursive_old_273015ac.json` (old) and `0003_recursive_engine_d097dbda.json` (new). Compare
them with `pytest-benchmark compare` or `pytest --benchmark-compare=0002 --benchmark-only`.

| Benchmark (`test_recursive_apply_<name>_benchmark`) | Input                                                                       | Old median | New median | Speed-up | Old min  | New min  | Speed-up (min) |
| --------------------------------------------------- | --------------------------------------------------------------------------- | ---------- | ---------- | -------- | -------- | -------- | -------------- |
| `scalar`                                            | `1`                                                                         | 0.7 µs     | 0.6 µs     | 1.27x    | 0.7 µs   | 0.5 µs   | 1.25x          |
| `empty_list`                                        | `[]`                                                                        | 1.0 µs     | 1.2 µs     | 0.82x    | 0.8 µs   | 1.0 µs   | 0.80x          |
| `small_dict`                                        | `{"a": 1, "b": [2, 3], "c": {"d": 4}}`                                      | 4.9 µs     | 4.4 µs     | 1.11x    | 4.7 µs   | 4.2 µs   | 1.12x          |
| `deep_list[10]`                                     | list nested 10 times                                                        | 8.0 µs     | 7.8 µs     | 1.03x    | 7.8 µs   | 7.5 µs   | 1.05x          |
| `deep_list[100]`                                    | list nested 100 times                                                       | 79.5 µs    | 72.5 µs    | 1.10x    | 78.3 µs  | 70.8 µs  | 1.11x          |
| `deep_list[300]`                                    | list nested 300 times                                                       | 247.7 µs   | 214.0 µs   | 1.16x    | 244.2 µs | 210.5 µs | 1.16x          |
| `tree[4-10]`                                        | depth 4, width 10 (10^4 leaves)                                             | 6.09 ms    | 1.67 ms    | 3.64x    | 6.06 ms  | 1.67 ms  | 3.63x          |
| `tree[14-2]`                                        | depth 14, width 2 (2^14 leaves)                                             | 23.07 ms   | 16.55 ms   | 1.39x    | 22.67 ms | 16.08 ms | 1.41x          |
| `tree[10-3]`                                        | depth 10, width 3 (59,049 leaves)                                           | 56.09 ms   | 30.83 ms   | 1.82x    | 55.43 ms | 30.48 ms | 1.82x          |
| `flat_list_identity`                                | list of 100,000 ints                                                        | 50.40 ms   | 7.93 ms    | 6.36x    | 50.16 ms | 7.83 ms  | 6.41x          |
| `flat_list_of_strings`                              | list of 100,000 `str`                                                       | 51.42 ms   | 8.02 ms    | 6.41x    | 51.17 ms | 7.91 ms  | 6.47x          |
| `flat_list`                                         | list of 100,000 ints                                                        | 52.26 ms   | 9.78 ms    | 5.35x    | 52.06 ms | 9.52 ms  | 5.47x          |
| `flat_tuple`                                        | tuple of 100,000 ints                                                       | 52.42 ms   | 9.79 ms    | 5.36x    | 52.29 ms | 9.47 ms  | 5.52x          |
| `flat_set`                                          | set of 100,000 ints                                                         | 53.93 ms   | 10.50 ms   | 5.14x    | 53.60 ms | 10.39 ms | 5.16x          |
| `flat_dict`                                         | dict of 100,000 ints                                                        | 58.13 ms   | 14.42 ms   | 4.03x    | 57.50 ms | 14.27 ms | 4.03x          |
| `list_of_small_dicts`                               | 10,000 dicts of an int, a list and a tuple                                  | 50.62 ms   | 29.10 ms   | 1.74x    | 50.28 ms | 28.63 ms | 1.76x          |
| `mixed_containers`                                  | list, tuple, `deque`, set, `frozenset`, named tuples and `str`, 10,000 each | 50.55 ms   | 15.29 ms   | 3.31x    | 50.14 ms | 14.86 ms | 3.37x          |

Reading the numbers:

- **The engine is 4 to 6 times faster on flat containers.** 100,000 leaves went from about 51-58 ms
  to 8-14 ms (speed-up 4.0x to 6.4x), about 0.1 µs per node instead of 0.5 µs. The constraint "no
  slower than today" holds with a large margin.
- **Trees gain less (1.4x to 3.6x)** because they are dominated by container creation and by many
  small containers: `tree[10-3]` (59,049 leaves) went from 56 ms to 31 ms, `tree[14-2]` from 23 ms
  to 17 ms, `list_of_small_dicts` from 51 ms to 29 ms.
- **Small inputs are about the same.** A leaf costs 0.6 µs instead of 0.7 µs, a small nested dict
  4.4 µs instead of 4.9 µs, and depth is still linear (0.7 µs per level, speed-up 1.03x to 1.16x).
  The one regression is the empty list (1.0 µs to 1.2 µs, speed-up 0.82x, about 0.2 µs per call),
  the fixed cost of setting up the engine and its per-call cache.
- A list nested about a thousand times raised `RecursionError` on the old implementation (F2), so
  the benchmark stops at 300. Add deeper cases (for example 10,000 levels) now that the explicit
  stack has landed; they cannot run on the old design.

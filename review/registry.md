# `coola.registry`: target state and the changes to reach it

This document describes the design that `coola.registry` should have, then the changes to make to
the current implementation (`main`, `5ce6dfcd`) to get there. The design follows the rules of the
user guide (`docs/docs/uguide/registry.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: registry dispatch, MRO lookup, lazy imports, strict contracts,
package layering: `registry` is a **core** package that imports no feature package), and it serves
the five packages built on it (`equality`, `hashing`, `summary`, `recursive`, `iterator`) and
`random`. Its aim is a small package that is easy to maintain: **one storage class, one dispatch
class, one way to match a type that cannot be a key**. Breaking changes are accepted, and the future
state is more important than the past one.

Evidence comes from probes and a throw-away prototype run on `main` (Python 3.14, scripts kept
outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from the
code. Nothing in `src`, `tests` or `docs` was changed.

## 1. Properties of the sub-package

These are the rules of the user guide, completed with the rules that the code already follows but
the guide does not state (fork, lazy loading, copy). Every design choice below exists to make one
of them true **by construction** instead of by the discipline of each registry.

| Property                     | Statement                                                                                                                                                                                                                                                                                             | Enforced by                                    |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Most specific type wins      | `TypeRegistry.resolve(t)` returns the handler of the first registered class of `t.__mro__`. The order of precedence is fixed: **registered types (MRO, `object` excluded) > fallbacks (registration order) > `object` > `TypeNotRegisteredError`**.                                                   | one lookup function                            |
| Extensible, no subclassing   | A type is supported by registering one handler. A type that cannot be a key (every dataclass, the classes that define a method, the virtual subclasses of an ABC) is supported by registering a **fallback predicate**.                                                                               | `register`, `register_fallback`                |
| No silent overwrite          | Registering an existing key (a type, a fallback predicate, or an entry of a lazy group) raises `AlreadyRegisteredError`, unless `exist_ok=True`. `register_many` is atomic: nothing is registered if one key is invalid or a duplicate.                                                               | the write path                                 |
| Thread-safe, cheap reads     | A reader sees the state before or after a mutation, never a partial one. A lookup of a known type takes **no lock**. The writers are serialized by one lock.                                                                                                                                          | state replaced, never mutated                  |
| Stable views                 | `items()`, `keys()`, `values()`, `fallbacks()` and iteration are snapshots: a later registration does not change a view already returned, and registering while iterating is safe.                                                                                                                    | state replaced, never mutated                  |
| Independent copies           | `copy()` returns a registry of the same class with the same entries, fallbacks and pending lazy groups. Registering in one does not change the other. The handlers are shared, not copied.                                                                                                            | `copy`                                         |
| Fork-safe and picklable      | A child created by `os.fork` never blocks on a lock held by a thread that does not exist there. A registry (with picklable handlers and predicates) survives `pickle` and `copy.deepcopy`, so it can be sent to a spawned worker.                                                                     | `track_fork`, `__getstate__`                   |
| Lazy for optional libraries  | Registering the handlers of an optional library imports nothing. A group is loaded when a type from its package is looked up, or when the whole content is read. `repr` and `copy` never load. A group that fails to load stays pending and raises again. Loaded entries never replace explicit ones. | one lazy mechanism in `TypeRegistry`           |
| Strict keys and clear errors | `TypeRegistry` keys are classes (`TypeError` otherwise). A missing key raises `KeyNotRegisteredError`, a missing type `TypeNotRegisteredError`; both are `KeyError`. The message lists the MRO of the type and the registered types.                                                                  | `_validate_key`, the message builder           |
| Insertion order              | `items()`, `keys()`, `values()`, iteration and `fallbacks()` follow the order of registration (`RandomManagerRegistry` relies on it).                                                                                                                                                                 | `dict`                                         |
| Generic typing               | `Registry[K, V]` and `TypeRegistry[V]` are generic.                                                                                                                                                                                                                                                   | `Generic`                                      |
| Core package                 | `coola.registry` imports no feature package (no deferred import either) and no optional library. It depends on `coola.utils` only (`fork`, `format`, `imports`).                                                                                                                                      | import-linter, **without** an `ignore_imports` |
| One display                  | Every registry prints the same way: `ClassName(` then one `(key): value` line per entry, then the fallbacks and the pending lazy groups.                                                                                                                                                              | one `__repr__`                                 |

Not guaranteed, and documented as such:

- **Virtual subclasses.** The MRO lookup does not call `issubclass`: a class that is only
  _registered_ with an ABC (`deque` with `Sequence`) is not matched by a handler registered for the
  ABC. Register the class, or register a fallback `lambda t: issubclass(t, Sequence)`.
- **The handlers.** They are stored as given: not copied, not checked against a protocol.
- **The predicates of the fallbacks** must be pure and cheap: the result is cached per type until
  the next change of the registry. A predicate that raises propagates and is not cached.
- **A registration made while another thread resolves** is seen by the lookups that start after the
  registration returns. A traversal that already holds a handler (the per-call caches of the
  engines) keeps it until the end of the call.
- **Memory of the cache.** The cache holds at most 1,024 classes and is emptied by any change.

## 2. What differs today

Constraints that framed the analysis:

1. **Simple and easy to maintain.** The package is 1,550 lines. 693 lines are in `base.py` for a class
   (`Registry`) that adds no code on top of it; 308 lines are in `dispatch.py`, 250 of them are
   delegations to `TypeRegistry`. The target removes the layers that exist only to work around the
   design.
2. **Major release.** Compatibility is not a design input: no adapter, no alias, no deprecation
   period. The release notes list the changes.
3. **No slower than today**, measured by the benchmark gate of section 8. The registry is on the hot
   path of `hash_object` (one lookup per node), of `summarize`, and of the first node of each type of
   the other engines.
4. **Consistent with the siblings.** Five packages subclass the dispatch registry. A change of the
   contract is a change of the five; section 5 lists them.
5. **Optional libraries stay lazy.** `import coola.registry` imports no optional library (it takes
   18 ms, 8.5 ms of it for `coola.display` **[verified]**; `tests/integration/test_lazy_imports.py`
   guards the rest).

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Status         |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The cache is slower than the lookup it caches.** A cached `resolve` costs 208-224 ns whatever the MRO; a `dict.get` costs 31 ns; the uncached MRO walk costs 100-156 ns. The cost is the `WeakKeyDictionary`: it builds a weak reference per lookup. On a miss the cache is **copied** (copy on write, up to 1,024 entries): resolving 5,000 distinct types takes 0.35 s (72 µs per miss).                                                                                                                                                                                                                                                                                 | **[verified]** |
| F2  | **Two mechanisms for "a type that cannot be a key".** `DataclassMarker` (equality, summary, iterator, recursive) is a fake type used as a key: it is visible in `items()`, `len()` and `copy()`, it needs a hook (`_accepts_dataclass_marker`, with a `# noqa: ARG002`), and a lookup that ends on `object` costs four `find` calls plus `is_dataclass` (1,187 ns per `find_transformer` on a dataclass). `HasherRegistry.register_fallback` (predicate based) does the same job with its own lock, `copy`, `clear`, `equal`, `_display_fields` and fork code (about 110 lines) and reaches into the private `self._state.resolve_or` and `self._state._not_registered_msg`. | **[verified]** |
| F3  | **The marker is chosen by identity.** `_find_with_dataclass_marker` uses the marker handler when `handler is self.find(object)`. If the same instance is registered for `object` and for a base class `Base`, a dataclass that inherits `Base` gets the marker handler instead of the `Base` handler (`SummarizerRegistry` with one `DefaultSummarizer` shared: the lookup returns `DataclassSummarizer`; with two instances it returns `DefaultSummarizer`).                                                                                                                                                                                                                | **[verified]** |
| F4  | **`register_lazy` breaks when a name is reused.** The groups are stored by name, with one entry per alias. `register_lazy(["builtins", "decimal"], f)` then `register_lazy("builtins", g)` replaces the alias `builtins` only: the next load raises `KeyError: 'builtins'` (the second alias still points to the first group). With one name, the second call silently drops the first group.                                                                                                                                                                                                                                                                                | **[verified]** |
| F5  | **The lazy mechanism is generic but only meaningful for types.** `BaseRegistry` carries `_LazyEntry`, `_lazy`, `_lazy_names`, `_load_lazy*`, `_pending_msg` (about 130 lines); `Registry` never uses them (their default `_lazy_names` returns every group for every key); the three callers pass top-level package names matched against `__module__`.                                                                                                                                                                                                                                                                                                                      | **[reading]**  |
| F6  | **Locking and derived state for nothing.** Every read (`has`, `get`, `[]`, `len`, `iter`) takes an `RLock` (`has` costs 182 ns); `_snapshot`, `_get_snapshot`, `_invalidate` and `_on_change` exist to keep a cache and a snapshot in sync with a state that is mutated in place. Replacing the state instead of mutating it removes the four.                                                                                                                                                                                                                                                                                                                               | **[reading]**  |
| F7  | **Three layers for two behaviours.** `BaseRegistry` → `Registry` (an empty subclass: 71 lines of docstring) and `TypeRegistry`; then `BaseTypeDispatchRegistry` wraps a `TypeRegistry` by **composition** and redeclares `register`, `register_many`, `register_lazy`, `has`, `find`, `unregister`, `clear`, `items`, `equal`, `copy`, `__contains__`, `__len__`. The same operation has two names (`find` and `resolve`), and a third pair for each domain (`find_hasher`, ...).                                                                                                                                                                                            | **[reading]**  |
| F8  | **Two display formats.** `Registry` and `TypeRegistry` print `TypeRegistry(\n  (<class 'int'>): i\n)`; the dispatch registries print `HasherRegistry(\n  (state): TypeRegistry(\n      (<class 'int'>): StrHasher()\n    )\n)` (a wrapper level, 6 spaces, and a blank line when empty). Four docstrings (hashing, equality, summary, recursive) quote the second format.                                                                                                                                                                                                                                                                                                    | **[verified]** |
| F9  | **`equal()` is wrong for two of the six default registries and costs the only layering exception.** `get_default_registry().copy().equal(get_default_registry())` is `False` for the hashing and summary registries (handlers have no `__eq__`, `objects_are_equal` returns `False` for two equal-by-identity handlers), `True` for the four others. Nothing in `src` calls it except the override of `HasherRegistry`. It imports `coola.equality` at call time (the only `ignore_imports` of import-linter, "core must not import features"), holds two locks and orders them by `id`. Decision D4: `equal` is kept and rewritten (section 3.8).                                                                                     | **[verified]** |
| F10 | **Not picklable.** `pickle.dumps(Registry(...))`, `pickle.dumps(TypeRegistry(...))` and `copy.deepcopy(...)` raise `TypeError: cannot pickle '_thread.RLock' object`. A registry cannot be passed to a `spawn` worker (the default start method on macOS and Windows) or be part of a deep-copied configuration.                                                                                                                                                                                                                                                                                                                                                             | **[verified]** |
| F11 | **Compatibility baggage.** `AlreadyRegisteredError(ValueError, RuntimeError)` "which this error used to be"; `KeyNotRegisteredError` documented as "so existing `except KeyError` code keeps working"; the property "Backward-compatible errors" of the guide; `register(key, value, exist_ok=False)` takes `exist_ok` positionally while `CLAUDE.md` asks for keyword-only options; `resolve_or` is public only for the hashing registry.                                                                                                                                                                                                                                   | **[verified]** |
| F12 | **The guide contradicts the code.** The two `# doctest: +SKIP` examples show `RuntimeError: A value is already registered ...` (the class is `AlreadyRegisteredError`); the warning about abstract base classes sits under "Registry (Generic Key-Value)" and uses `TypeRegistry`; "See also" says `coola.recursive` "uses `TransformerRegistry` (a `TypeRegistry`)" (it is a `BaseTypeDispatchRegistry` that contains one); `register_lazy`, `DataclassMarker`, fallbacks and `BaseTypeDispatchRegistry` are not documented outside `architecture.md`; the Properties table omits fork, lazy and copy.                                                                      | **[verified]** |
| F13 | **`TypeRegistry.get(3)` returns `None`** while `register(3, ...)` and `resolve(3)` raise `TypeError`. A mapping lookup of a non-key is allowed to return the default (`{}.get(3)`); this is kept and documented, not changed.                                                                                                                                                                                                                                                                                                                                                                                                                                                | **[verified]** |

What is already right and must not regress: MRO lookup with `object` as catch-all and the explicit
note about ABCs; the atomic `register_many`; `exist_ok`; insertion order; the error messages of
`TypeNotRegisteredError` (MRO and registered types, with the module added to ambiguous names);
`repr` that does not import the optional libraries; lazy entries that never replace explicit
ones; a lazy group that fails stays pending; the reentrant factory (a lazy factory may register in
the registry that loads it **[verified]**); `copy()` that shares the handlers; the fork reset;
`KeyError` as the base of the lookup errors; 100 % coverage of the package (462 tests pass on
`main`).

## 3. Architecture

```
exceptions.py   AlreadyRegisteredError(ValueError)
                KeyNotRegisteredError(KeyError) <- TypeNotRegisteredError

registry.py     Registry[K, V]                 storage: one dict, replaced on every write
                |   register, register_many, unregister, clear, copy
                |   get, has / in, [], len, iter, keys, values, items, repr
                |   writer lock, track_fork, pickle
                v
type.py         TypeRegistry[V](Registry[type, V])      dispatch
                |   resolve(dtype)       cache -> MRO (no object) -> fallbacks -> object
                |   register_fallback / unregister_fallback / fallbacks()
                |   register_lazy       groups of handlers for optional packages
                v
domain          HasherRegistry, EqualityTesterRegistry, SummarizerRegistry,
                ChildFinderRegistry, TransformerRegistry     (subclasses of TypeRegistry)
                |   has_<x>, find_<x> (one line each), the domain entry point
```

`BaseRegistry`, `BaseTypeDispatchRegistry` and `DataclassMarker` disappear; `vanilla.py` and
`dispatch.py` are deleted. Two classes remain. A domain registry **is** a `TypeRegistry` (it
inherits, it does not wrap), so it has no `_state` wrapper and one display.

### 3.1 State: replaced, never mutated

```python
class Registry(Generic[K, V]):
    def __init__(self, initial_state=None):
        self._state: dict[K, V] = dict(initial_state or {})  # never mutated after publication
        self._lock = threading.RLock()  # writers only
        track_fork(self)

    def register(self, key, value, *, exist_ok=False):
        self._validate_key(key)
        with self._lock:
            self._load_for(key)  # a lazy default is seen by the exist_ok check
            if key in self._state and not exist_ok:
                raise AlreadyRegisteredError(self._already_registered_msg(key))
            self._publish({**self._state, key: value})

    def items(self):  # a view of a dict that is never mutated is a snapshot
        self._load_all()
        return self._state.items()
```

- **Readers take no lock**: they read `self._state` once. A mutation builds a new dict (`O(n)`, `n`
  is about 50 and a registration is rare) and swaps the reference, which is atomic. This replaces
  `_snapshot`, `_get_snapshot`, `_invalidate`, `_on_change` and the per-method `with self._lock`.
- `register_many` builds one dict for the whole mapping and publishes it once: atomic by
  construction, including for readers.
- `_publish(state)` assigns `_state` and calls `_changed()`, a hook that `TypeRegistry` overrides to
  empty the cache. Three hooks exist for the subclass, all no-ops in `Registry`: `_validate_key`,
  `_load_for(key)`, `_load_all()`. (Today: `_validate_key`, `_on_change`, `_lazy_names` and
  the three `_load_lazy*`.)
- `exist_ok` is keyword-only everywhere (`register`, `register_many`, `register_fallback`).

### 3.2 Fork and pickle

`_reset_after_fork` recreates the (only) lock, as today. `__getstate__` returns `__dict__` without
`_lock` and `_cache`; `__setstate__` recreates them and calls `track_fork`. Ten lines in
`Registry`; `TypeRegistry` adds nothing because its other attributes are immutable containers.
A registry holding lambdas (a predicate, a lazy factory) is not picklable, like any object holding
a lambda; the default registries hold module-level functions and `"module:name"` paths.

### 3.3 Dispatch

```python
def resolve(self, dtype: type) -> V:
    try:
        return self._cache[dtype]  # lock-free hit
    except KeyError:
        pass
    with self._lock:
        self._validate_key(dtype)
        self._load_for(dtype)
        value = self._lookup(dtype)  # raises TypeNotRegisteredError
        if len(self._cache) >= _MAX_CACHE_SIZE:
            self._cache = {}
        self._cache[dtype] = value  # a miss is not cached: _lookup raised
        return value


def _lookup(self, dtype):
    state = self._state
    for base in dtype.__mro__[:-1]:  # most specific first; object is the last resort
        if base in state:
            return state[base]
    for predicate, handler in self._fallbacks.items():  # registration order
        if predicate(dtype):
            return handler
    if object in state:
        return state[object]
    raise TypeNotRegisteredError(self._not_registered_msg(dtype))
```

- A plain `dict` replaces the `WeakKeyDictionary` and the copy on write (F1). The miss path is
  under the lock, as the registration is, so a stale value cannot be written after a registration
  has emptied the cache. The cache is bounded (1,024) and emptied by any change (`register`,
  `unregister`, `clear`, `register_fallback`, a lazy load).
- The fallbacks sit between the types and `object`. This is the rule of `DataclassMarker` today
  ("used only if the lookup ended on `object`") without the identity test (F3), and it keeps the
  rule of `HasherRegistry` ("a type registered always wins"), except for the corner case "a hasher
  registered for `object` makes the fallbacks unreachable": `object` is now the **last** resort
  everywhere. `resolve(object)` checks the fallbacks too (no predicate of the repository accepts
  `object`).
- `resolve_or` and `find` are deleted: the former exists to skip the message of the error for a
  fallback (the fallbacks are now in the lookup), the latter is a second name of `resolve`.

### 3.4 Fallbacks

`TypeRegistry._fallbacks` is a `dict[Callable[[type], bool], V]`, replaced on write like `_state`.

| Method                                                     | Behaviour                                                                                                                                     |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `register_fallback(predicate, handler, *, exist_ok=False)` | Appends the pair. A predicate already registered raises `AlreadyRegisteredError` (today: a duplicate is silently appended and never reached). |
| `unregister_fallback(predicate) -> handler`                | Removes the pair and returns the handler; `KeyNotRegisteredError` (a `KeyError`) if absent.                                                   |
| `fallbacks() -> ItemsView`                                 | The `(predicate, handler)` pairs in order, as a snapshot.                                                                                     |
| `clear()`                                                  | Removes the types, the fallbacks and the pending lazy groups. `copy()` and `repr` include the fallbacks.                                      |

The dataclass rule of each package becomes one registration in its `get_default_registry`:

```python
registry.register_fallback(dataclasses.is_dataclass, DataclassSummarizer())
# equality: the dataclass must use the __eq__ generated by @dataclass
registry.register_fallback(_has_generated_eq, DataclassEqualityTester())
# hashing: unchanged
registry.register_fallback(dataclasses.is_dataclass, DataclassHasher())
registry.register_fallback(_has_hash_method, HashableHasher())
```

### 3.5 Lazy groups (`TypeRegistry` only)

```python
@dataclass(frozen=True)
class _LazyGroup:
    packages: frozenset[str]
    factory: str | Callable[[], Mapping[type, V]]  # "package.module:name" is imported at load


self._lazy: tuple[_LazyGroup, ...] = ()  # replaced on write
```

- `register_lazy(packages, factory)` appends a group. **Two groups with the same package both
  stay** (F4); there is no alias table, so there is nothing to keep in sync.
- `_load_for(dtype)`: if `self._lazy` is empty, return (the common case after warm-up). Otherwise
  compute the top-level modules of the classes of `dtype.__mro__` and load the groups whose
  `packages` intersect them. `_load_all()` loads every group (`len`, iteration, `items`, `keys`,
  `values`).
- Loading runs under the writer lock: remove the group, call the factory, `setdefault` each entry
  after `_validate_key`, publish. If the factory raises, the group is put back and the error
  propagates (as today), so the next call fails the same way.
- A lazy factory may call `register` on the same registry (the lock is reentrant).
- `repr` lists the pending groups by package name and loads nothing; `copy()` copies the tuple and
  loads nothing.
- `register(key, ...)`, `unregister(key)`, `has(key)`, `get(key)` and `[key]` call `_load_for(key)`,
  so `has(np.ndarray)` does not depend on whether numpy was already looked up. Registering over a
  lazy default therefore needs `exist_ok=True` **[verified]**; this stays and is documented.

### 3.6 Errors

```python
class KeyNotRegisteredError(KeyError):  # __str__ is not repr()'d, as today
class TypeNotRegisteredError(KeyNotRegisteredError): ...
class AlreadyRegisteredError(ValueError): ...  # RuntimeError is dropped
```

The message builders stay (`Type '<class ...>' is not registered (MRO: ...). Registered types:
...`), as module-level functions of `type.py` because the hashing registry no longer calls them.

### 3.7 What a domain registry looks like

```python
class HasherRegistry(TypeRegistry[Hasher[Any]]):
    def has_hasher(self, data_type: type) -> bool:
        return self.has(data_type)

    def find_hasher(self, data_type: type) -> Hasher[Any]:
        return self.resolve(data_type)

    def hash(self, data, length=64, ignore_unhashable=False): ...  # unchanged
```

`HasherRegistry` loses `__init__`, `copy`, `_reset_after_fork`, `_display_fields`,
`register_fallback`, `unregister_fallback`, `clear` and `equal` (about 110 lines; `equal` is inherited, see 3.8): the base class
does them. `EqualityTesterRegistry` loses `_accepts_dataclass_marker`.

### 3.8 `equal` (decision D4: kept)

`equal(other)` stays, rewritten so that it is right and needs no feature package:

```python
def equal(self, other: object) -> bool:
    if type(other) is not type(self):
        return False
    self._load_all()
    other._load_all()
    return self._state == other._state and self._fallbacks == other._fallbacks
```

- The states are replaced, never mutated, so the comparison needs **no lock** (the two-lock,
  `id`-ordered code and the deferred `coola.equality` import are deleted, and with them the
  import-linter exception).
- Handlers are compared with `==`: a handler without `__eq__` is equal to itself only. A `copy()` is
  therefore equal to its source (the handlers are shared), which fixes F9 for the hashing and summary
  registries. Two registries built separately with handlers of the same class are equal only if the
  handlers define `__eq__`.
- `equal_nan` is dropped (it had no meaning for `==`). `HasherRegistry.equal` is deleted: the base
  compares the fallbacks too.

## 4. Public API after the change

```python
from coola.registry import (
    Registry,  # Registry[K, V]
    TypeRegistry,  # TypeRegistry[V]
    AlreadyRegisteredError,
    KeyNotRegisteredError,
    TypeNotRegisteredError,
)
```

| Class          | Methods                                                                                                                                                                                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Registry`     | `register(key, value, *, exist_ok=False)`, `register_many(mapping, *, exist_ok=False)`, `unregister(key) -> value`, `clear()`, `copy()`, `get(key, default=None)`, `has(key)`, `[]`, `[]=`, `del`, `in`, `len`, `iter`, `keys()`, `values()`, `items()`, `equal(other)` |
| `TypeRegistry` | all of `Registry`, plus `resolve(dtype)`, `register_fallback(predicate, handler, *, exist_ok=False)`, `unregister_fallback(predicate)`, `fallbacks()`, `register_lazy(packages, factory)`                                                               |

Removed: `BaseTypeDispatchRegistry`, `DataclassMarker`, `find`, `resolve_or`, the `equal_nan` argument of `equal`,
the `RuntimeError` base of `AlreadyRegisteredError`, the positional `exist_ok`, `_snapshot` and the
other private hooks listed in 3.1. Changed: the display of the dispatch registries, the order of
precedence of `object` against the fallbacks of the hashing registry.

## 5. Dependents

| Package / file                                | Change                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `equality/tester/registry.py`, `interface.py` | `EqualityTesterRegistry(TypeRegistry[...])`; delete `_accepts_dataclass_marker` and the `DataclassMarker` export; the default registry calls `register_fallback(_has_generated_eq, DataclassEqualityTester())`, where `_has_generated_eq` is the current condition (`__dataclass_params__.eq and _has_generated_eq`). `find_equality_tester` is `resolve`. |
| `hashing/registry.py`, `interface.py`         | `HasherRegistry(TypeRegistry[...])`; delete the fallback code, `copy`, `clear`, `equal`, `_display_fields`, the lock and the access to `_state`; keep `_find_enum_hasher`, `hash` and the thread-local cycle state (outside this design). `register_fallback` keeps its signature (+ `exist_ok`); `unregister_fallback` now returns the hasher.            |
| `summary/registry.py`, `interface.py`         | Same as equality, with `dataclasses.is_dataclass` as the predicate.                                                                                                                                                                                                                                                                                        |
| `iterator/registry.py`, `interface.py`        | Same, `DataclassChildFinder`. `enter_dataclasses` (opt-in) registers the fallback instead of the marker; its fate is decided by `review/iterator.md` and `review/recursive.md` (D1 there).                                                                                                                                                                 |
| `recursive/registry.py`, `interface.py`       | Same, `DataclassTransformer`. `review/recursive.md` (step 5, "dataclasses by default") must read "register the fallback `dataclasses.is_dataclass`" where it says "register `DataclassTransformer` on `DataclassMarker`". `recursive/dataclass.py` and the guide mention the marker.                                                                       |
| `nested/conversion.py`                        | `registry.register(DataclassMarker, DefaultTransformer(), exist_ok=True)` becomes `registry.register_fallback(dataclasses.is_dataclass, DefaultTransformer(), exist_ok=True)`: a dataclass is a leaf for `to_jsonable`.                                                                                                                                    |
| `random/registry.py`                          | `RandomManagerRegistry(Registry[str, BaseRandomManager], BaseRandomManager)` is unchanged; it gains pickling and `exist_ok` as keyword-only (check the callers).                                                                                                                                                                                           |
| `utils/singleton.py`, `testing/registries.py` | `temporary_default_registries` uses `copy()` and `_replace_with_copy`; no change of code, the docs say what a copy contains (fallbacks).                                                                                                                                                                                                                   |
| `pyproject.toml` (import-linter)              | Delete `ignore_imports = ["coola.registry.base -> coola.equality"]` (F9). `docs/docs/dev/architecture.md` ("Package Layering") loses the sentence about the exception.                                                                                                                                                                                     |
| Tests that read `registry._state`             | `tests/unit/recursive/test_registry.py`, `tests/unit/hashing/test_registry.py`: `_state` is still a `dict` of types to handlers, so they keep running, but they should use the public `registry[list]`, `list in registry`, `len(registry)`.                                                                                                               |
| Docs                                          | `uguide/registry.md` (rewrite), `dev/architecture.md` (registry paragraph, layering), `dev/add_backend.md` (`register_lazy`), the guides of `hashing`, `equality_tester`, `summary`, `iterator`, `recursive` (fallbacks, marker), the migration page of the next major version.                                                                            |

## 6. Evidence: a prototype of the lookup

A prototype of 3.1-3.4 (a dict state replaced on write, a plain `dict` cache, `object` as last
resort, fallbacks in the lookup; no lazy groups) was run on `main` against the current
`TypeRegistry` with the same entries (`object`, `int`, `Mapping`, `list`, `dict`). It is not part of
the repository. **[verified]**

| Input                                                                 | `main`   | Prototype          |
| --------------------------------------------------------------------- | -------- | ------------------ |
| `resolve(bool)`, MRO of 3, cached                                     | 213 ns   | 45 ns              |
| `resolve(list)`, MRO of 2, cached                                     | 217 ns   | 48 ns              |
| `resolve(D)`, MRO of 5, cached                                        | 210 ns   | 46 ns              |
| `resolve(OrderedDict)`, cached                                        | 215 ns   | 45 ns              |
| `find_transformer(a_dataclass)` (marker on `main`, fallback in proto) | 1,187 ns | 47 ns              |
| 5,000 distinct types resolved once                                    | 0.353 s  | 0.002 s            |
| `resolve` with **no cache** at all (variant, MRO of 2 / 3 / 5)        | n/a      | 117 / 130 / 246 ns |

Reading the numbers:

- The cache is worth keeping, but as a plain `dict`. Without any cache the MRO walk is 117-246 ns,
  about the cost of today's cache hit, and every dataclass would run its predicates on every node
  (hashing looks up every node). With the plain `dict`, a hit is 4.5 times faster than today.
- `main` already bounds its cache; what it pays for is the `WeakKeyDictionary` (a weak reference per
  lookup, about 180 ns) and the copy on write.
- The pickle and `deepcopy` round trip of the prototype (state, no lock) works; the real one is
  verified in step 6.
- `has`, `get` and `len` lose their `RLock` (182 ns for `has` today). The gain is expected to be
  about the cost of the lock; it is measured by the benchmark gate, not claimed here **[reading]**.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable and
puts the independent, low-risk steps first.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Breaking              |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| 1    | **Fix the guide (F12).** `AlreadyRegisteredError` in the two SKIP'd examples, the ABC warning under `TypeRegistry`, the "See also" paragraph, add `register_lazy`, `copy`, fork, and the **Properties** table of section 1 at the top of the page and in the package docstring. Documentation only.                                                                                                                                                                                                                 | no                    |
| 2    | **Plain `dict` cache (F1).** Replace the `WeakKeyDictionary` copy on write by a `dict` filled under the lock, bounded to 1,024, emptied by `_on_change`. Replace `test_type_registry_cache_does_not_keep_a_dynamic_class_alive` by a test that the cache is bounded. Benchmarks: cached hit, 5,000 types. Independent of the other steps.                                                                                                                                                                           | no                    |
| 3    | **Fallbacks in `TypeRegistry` (F2, F3).** Add `register_fallback`, `unregister_fallback`, `fallbacks()` to `TypeRegistry`, the lookup order of 3.3, `copy`/`clear`/`repr` support. Then one PR per package moves it from the marker (equality, summary, iterator, recursive, `nested.conversion`) or from its own code (hashing) to the base fallbacks. Last PR: delete `DataclassMarker`, `_find_with_dataclass_marker`, `_accepts_dataclass_marker`, `resolve_or`. Regression test for F3 (fails on `main`).      | yes (marker, display) |
| 4    | **One storage class, replaced state (F5 partly, F6, F7, F8, F9, F11).** Merge `BaseRegistry` and `Registry`; `TypeRegistry` extends it; the domain registries extend `TypeRegistry` and `BaseTypeDispatchRegistry` is deleted; the state is replaced on write; reads take no lock; one `__repr__`; `exist_ok` keyword-only; rewrite `equal` without the `equality` import (D4, 3.8), delete `find`, `AlreadyRegisteredError`'s `RuntimeError` base; remove the import-linter exception. Update the four docstrings that quote the old display. Benchmark gate. | yes                   |
| 5    | **Lazy groups in `TypeRegistry` as a tuple (F4, F5).** Move the lazy code from `Registry` to `TypeRegistry`, store the groups in a tuple, regression tests for both cases of F4 (they fail on `main`), keep the tests of reentrancy, failure and `repr`/`copy` not loading.                                                                                                                                                                                                                                         | no (a bug fix)        |
| 6    | **Pickle (F10).** `__getstate__` / `__setstate__`, a test with `pickle` and `copy.deepcopy` for both classes and for the default registries (all of them), and an integration test that sends a registry to a `spawn` process.                                                                                                                                                                                                                                                                                      | no                    |
| 7    | **Documentation.** Rewrite `uguide/registry.md` (properties, fallbacks, lazy groups, ABC fallback example, precedence table), the registry paragraphs of `dev/architecture.md` and the layering section, the migration page (marker, `find`, `resolve_or`, `equal_nan`, positional `exist_ok`, `RuntimeError`, display), and the status table of `review/suggestions.md`.                                                                                                                                               | no                    |

Steps 4 and 5 can be merged if the diff stays reviewable; step 4 first because it removes the code
that step 5 would otherwise have to change twice.

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/registry/test_registry.py`, `test_type.py`,
`test_exceptions.py`); the conventions of `CLAUDE.md` apply (one behaviour per test, literal
expected values, a regression test for every bug, no local imports). `dispatch.py`, `vanilla.py`,
`base.py` and their tests (about 1,500 lines of the 2,725) are replaced; most cases move to the
two remaining files unchanged.

- **`Registry`**: register, duplicate, `exist_ok`, `register_many` atomic (invalid key, duplicate,
  empty mapping), unregister returns the value, unregister of a missing key, clear, `copy`
  independent (both directions) and sharing the handlers, insertion order after overwrite and after
  unregister, `get` with a default, `in`, `[]`, `del`, **views are snapshots** (a registration after
  `items()` does not change the returned view; registering while iterating works), `repr` / `str`
  against a literal, the empty registry display, `exist_ok` is keyword-only (`TypeError`).
- **`TypeRegistry.resolve`**: exact match, parent, diamond MRO (the existing hierarchy), `object`
  fallback, miss (the whole message against a literal, with the ambiguous names), a non-type
  (`TypeError`), **precedence** (type > fallback > `object`, a table-driven parametrised test with
  all eight combinations), registration of a more specific type after a cached lookup is seen,
  `unregister` is seen, the bounded cache is emptied at 1,024, a miss is not cached, virtual
  subclasses are not matched (kept) **and** are matched by a fallback.
- **Fallbacks**: registration order, duplicate predicate (`AlreadyRegisteredError`), `exist_ok`
  keeps the position, `unregister_fallback` returns the handler and raises for an unknown
  predicate, a predicate that raises propagates and is not cached, `copy` / `clear` / `repr`
  include the fallbacks, a registered type beats a fallback and a fallback beats `object`, **a
  handler shared between `object` and a base class** (regression for F3).
- **Lazy groups**: a group loads when a type of its package is resolved, not for an unrelated type;
  `len`, `items`, `keys`, `values` and iteration load everything; `repr`, `copy` and `register_lazy`
  load nothing; a loaded entry never replaces an explicit one; `register` over a lazy default
  raises without `exist_ok`; a failing factory keeps the group and raises again; a factory that
  registers in the same registry; `"module:name"` imported at load; **two groups with one package
  and a group with two packages followed by a group with one of them (regression for F4)**; an
  invalid key returned by a factory.
- **Pickle and fork**: `pickle` and `copy.deepcopy` round trip (state, fallbacks, pending groups),
  the lock of the clone is its own, `_reset_after_fork` recreates the lock (existing test).
- **`equal`**: a registry equals its copy (including for each default registry), differs after a
  registration, an unregistration, a different handler, a different fallback or a different class,
  handlers compare with `==`, a pending lazy group is loaded before the comparison, no lock is held
  (a concurrent registration does not deadlock). Regression for F9: the hashing and summary default
  registries equal their copy (fails on `main`).
- **Errors**: the classes, the bases (`AlreadyRegisteredError` is not a `RuntimeError`), the
  `KeyError.__str__` override.
- **Contract test over the default registries** (`tests/integration/test_default_registries.py`,
  extend): for each of the six default registries `copy()` has the same keys and fallbacks, is
  independent, prints a string that starts with the class name, and pickles.
- **Properties** (`tests/properties/test_registry.py`, extend the state machine): the model gains
  fallbacks; `resolve` equals the model (types, then fallbacks, then `object`); a view taken before
  a rule is unchanged after it; `copy` is independent.
- **Integration** (`tests/integration/registry/`, `run_threads` from `tests/integration/helpers.py`,
  a lock around the shared list, docstrings starting with "Test that ..."): readers resolving while
  a writer registers see the old or the new handler only; `register_many` is atomic for readers; two
  threads resolving a never-seen type both succeed; concurrent `register_lazy` and loading; a spawn
  worker that receives a registry. No patching.
- **Doctests** run as part of the docstrings; the guide examples are doctests (the SKIP'd traceback
  examples become real ones with `except AlreadyRegisteredError`).
- **Benchmark gate** (`tests/benchmarks/test_registry_benchmark.py`, `inv test.benchmark`): keep
  `test_type_registry_resolve_cached_benchmark` and `..._uncached_benchmark` (it resets `_cache`
  with `{}`: still valid), add **a dataclass lookup through a fallback**, **5,000 distinct types**,
  **`has`** and **`register`** on a 50-entry registry; the new code must not be slower than `main`
  on any (expected: 4-5x faster for the cached hit and the dataclass, 100x or more for the cold
  types, `register` slower by one `O(n)` dict copy, about 1-2 µs, which is accepted).
- Coverage: 100 % of `src/coola/registry` (`--cov=coola.registry`), no `# pragma: no cover`.

## 9. Decisions for the maintainers

**D1. Replace `DataclassMarker` and the hashing fallbacks by one fallback mechanism?**
_Recommended: yes._ It is one idea implemented twice (F2), the marker's identity test is a bug
(F3), and a predicate also solves the virtual-subclass limit of the MRO lookup (a user can register
`lambda t: issubclass(t, Sequence)`). Consequences: `object` is the last resort in the hashing
registry too (nobody registers `object` there), and the dataclass handler is no longer a visible
"type" entry of `items()` / `len()`. Alternative: keep the marker and only fix the identity test;
cheaper, but the five registries keep a second mechanism and the hashing registry keeps its own.

**Decision: yes.**

**D2. Keep the generic `Registry`?** _Recommended: yes._ `RandomManagerRegistry` uses it and
`TypeRegistry` extends it, so it costs nothing: it is the storage class. Alternative: make it
private (`_Registry`) and drop it from the guide (the "plugin system", "factory" and "configuration"
examples are not what `coola` is for); this only changes the guide.

**Decision: yes.**

**D3. A strong bounded cache instead of a weak one?** _Recommended: yes._ The weak dictionary is the
cost (F1). The cache holds at most 1,024 classes and is emptied by any change, so a program that
creates classes dynamically keeps at most 1,024 of them alive (a few MB) until the next
registration, instead of the last reference. Alternative: no cache; simpler by about 12 lines, but
the dataclass fallback predicates run on every lookup (section 6).

**Decision: yes.**

**D4. Delete `equal()`?** _Recommended: yes. Answered: no, it is kept (rewritten as the alternative below, see 3.8)._ Nothing in `src` uses it, it is wrong for two default
registries (F9), and it is the only reason `registry` depends on `equality`. A test that needs to
compare two registries compares `dict(a.items()) == dict(b.items())`. Alternative: keep
`equal(other)` as `type(a) is type(b) and a._state == b._state and a._fallbacks == b._fallbacks`
(handlers compare by `==`, so a copy is equal), without `equal_nan` and without the second lock.

**Decision: no.** `equal` stays; it is rewritten without the `equality` import and without `equal_nan`.

**D5. Drop the `RuntimeError` base of `AlreadyRegisteredError`?** _Recommended: yes._ It is a
`ValueError` (the value is not acceptable in the current state). `except RuntimeError` code is
rare; the release notes say so. Alternative: keep it for one more major release.

**Decision: yes.**

**D6. Keep `resolve` as the only lookup name?** _Recommended: yes._ It is the name of the guide and
of `TypeRegistry`; `find` only exists in `BaseTypeDispatchRegistry`. The domain aliases `has_<x>` and
`find_<x>` stay (they read well at the call site and are public API of five packages): they become
one-line delegations. Dropping them is a change of five packages, not of this one.

**Decision: yes.**

**D7. Registries are picklable?** _Recommended: yes._ Ten lines, and it removes a failure that only
shows with `spawn`. Alternative: document that a registry is not picklable and pass the arguments
that build it.

**Decision: yes.**

**D8. A duplicate fallback predicate is an error?** _Recommended: yes_ (`exist_ok=True` to replace,
the position is kept). Today a duplicate is silently appended and never reached.

**Decision: yes.**

## 10. Not covered

- **`issubclass` lookup.** Not added to `resolve`: it would send `str` to the handler of `Sequence`
  (the reason in the current docstring). The fallbacks are the opt-in way.
- **Entry-point plugins** (24.3 of `suggestions.md`): how plugins are listed or disabled is a
  question of the packages that load them, not of the registry.
- **Scoped overrides** (`temporary_default_registries`): in `coola.utils.singleton`; unchanged.
- **Handler protocol check at registration.** The registry stays generic; the domain registries
  could validate their handlers, which is a separate item.
- **A registry shared between processes.** Out of scope; a copy per process is the model.
- **The per-call caches of the engines** (`recursive`, `iterator`, `equality`): they keep their
  design; this review makes the registry lookup they call cheaper but does not change them. The
  hashing registry has no per-call cache and benefits most.
- **The thread-local cycle state of `HasherRegistry.hash`**: tracked in `review/suggestions.md`
  (hashing), outside this design.

## Appendix: effect on the size of the package

| Module                                                    | Lines today  | Expected after | Why                                                                                              |
| --------------------------------------------------------- | ------------ | -------------- | ------------------------------------------------------------------------------------------------ |
| `base.py` (`BaseRegistry`) → `registry.py`                | 693          | about 330      | no snapshot, no `_invalidate` / `_on_change`, no lock on reads, a 6-line `equal`, no lazy code (moved) |
| `type.py`                                                 | 376          | about 400      | + fallbacks, + lazy groups (about 70 lines moved from `base.py`), − weak cache, − `resolve_or`   |
| `dispatch.py`                                             | 308          | 0              | deleted                                                                                          |
| `vanilla.py`                                              | 71           | 0              | deleted (`Registry` lives in `registry.py`)                                                      |
| `exceptions.py`                                           | 80           | about 70       | no compatibility paragraph                                                                       |
| `__init__.py`                                             | 22           | about 14       | five names                                                                                       |
| **`coola.registry`**                                      | **1,550**    | **about 815**  |                                                                                                  |
| `hashing/registry.py`                                     | 400          | about 290      | fallbacks, copy, clear, equal, display, fork, lock moved to the base                             |
| `equality`, `summary`, `iterator`, `recursive` registries | 4 × about 10 | −40            | `_accepts_dataclass_marker`, `DataclassMarker` import and export                                 |

The line counts are estimates from the prototype and from the code to delete (docstrings are most
of the lines); the gate is the behaviour (sections 1 and 8), not the size.

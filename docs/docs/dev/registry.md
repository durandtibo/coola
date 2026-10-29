# Architecture of `coola.registry`

:book: This page describes the design of the `coola.registry` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/registry.md#properties); the API is in the [reference](../refs/registry.md).

## Architecture

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

### State: replaced, never mutated

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

### Fork and pickle

`_reset_after_fork` recreates the (only) lock, as before. `__getstate__` returns `__dict__` without
`_lock` and `_cache`; `__setstate__` recreates them and calls `track_fork`. Ten lines in
`Registry`; `TypeRegistry` adds nothing because its other attributes are immutable containers.
A registry holding lambdas (a predicate, a lazy factory) is not picklable, like any object holding
a lambda; the default registries hold module-level functions and `"module:name"` paths.

### Dispatch

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

- A plain `dict` replaces the `WeakKeyDictionary` and the copy on write. The miss path is
  under the lock, as the registration is, so a stale value cannot be written after a registration
  has emptied the cache. The cache is bounded (1,024) and emptied by any change (`register`,
  `unregister`, `clear`, `register_fallback`, a lazy load).
- The fallbacks sit between the types and `object`. This is the rule of `DataclassMarker` today
  ("used only if the lookup ended on `object`") without the identity test, and it keeps the
  rule of `HasherRegistry` ("a type registered always wins"), except for the corner case "a hasher
  registered for `object` makes the fallbacks unreachable": `object` is now the **last** resort
  everywhere. `resolve(object)` checks the fallbacks too (no predicate of the repository accepts
  `object`).
- `resolve_or` and `find` are deleted: the former exists to skip the message of the error for a
  fallback (the fallbacks are now in the lookup), the latter is a second name of `resolve`.

### Fallbacks

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

### Lazy groups (`TypeRegistry` only)

```python
@dataclass(frozen=True)
class _LazyGroup:
    packages: frozenset[str]
    factory: str | Callable[[], Mapping[type, V]]  # "package.module:name" is imported at load


self._lazy: tuple[_LazyGroup, ...] = ()  # replaced on write
```

- `register_lazy(packages, factory)` appends a group. **Two groups with the same package both
  stay**; there is no alias table, so there is nothing to keep in sync.
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
  lazy default therefore needs `exist_ok=True`; this stays and is documented.

### Errors

```python
class KeyNotRegisteredError(KeyError):  # __str__ is not repr()'d, as before
class TypeNotRegisteredError(KeyNotRegisteredError): ...
class AlreadyRegisteredError(ValueError): ...  # RuntimeError is dropped
```

The message builders stay (`Type '<class ...>' is not registered (MRO: ...). Registered types:
...`), as module-level functions of `type.py` because the hashing registry no longer calls them.

### What a domain registry looks like

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

### `equal`

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
  therefore equal to its source (the handlers are shared), which fixes for the hashing and summary
  registries. Two registries built separately with handlers of the same class are equal only if the
  handlers define `__eq__`.
- `equal_nan` is dropped (it had no meaning for `==`). `HasherRegistry.equal` is deleted: the base
  compares the fallbacks too.

## Public API

```python
from coola.registry import (
    Registry,  # Registry[K, V]
    TypeRegistry,  # TypeRegistry[V]
    AlreadyRegisteredError,
    KeyNotRegisteredError,
    TypeNotRegisteredError,
)
```

| Class          | Methods                                                                                                                                                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Registry`     | `register(key, value, *, exist_ok=False)`, `register_many(mapping, *, exist_ok=False)`, `unregister(key) -> value`, `clear()`, `copy()`, `get(key, default=None)`, `has(key)`, `[]`, `[]=`, `del`, `in`, `len`, `iter`, `keys()`, `values()`, `items()`, `equal(other)` |
| `TypeRegistry` | all of `Registry`, plus `resolve(dtype)`, `register_fallback(predicate, handler, *, exist_ok=False)`, `unregister_fallback(predicate)`, `fallbacks()`, `register_lazy(packages, factory)`                                                                               |

Removed: `BaseTypeDispatchRegistry`, `DataclassMarker`, `find`, `resolve_or`, the `equal_nan` argument of `equal`,
the `RuntimeError` base of `AlreadyRegisteredError`, the positional `exist_ok`, `_snapshot` and the
other private hooks listed in 3.1. Changed: the display of the dispatch registries, the order of
precedence of `object` against the fallbacks of the hashing registry.

## Dependents

| Package / file                                | Change                                                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `equality/tester/registry.py`, `interface.py` | `EqualityTesterRegistry(TypeRegistry[...])`; delete `_accepts_dataclass_marker` and the `DataclassMarker` export; the default registry calls `register_fallback(_has_generated_eq, DataclassEqualityTester())`, where `_has_generated_eq` is the current condition (`__dataclass_params__.eq and _has_generated_eq`). `find_equality_tester` is `resolve`. |
| `hashing/registry.py`, `interface.py`         | `HasherRegistry(TypeRegistry[...])`; delete the fallback code, `copy`, `clear`, `equal`, `_display_fields`, the lock and the access to `_state`; keep `_find_enum_hasher`, `hash` and the thread-local cycle state (outside this design). `register_fallback` keeps its signature (+ `exist_ok`); `unregister_fallback` now returns the hasher.            |
| `summary/registry.py`, `interface.py`         | Same as equality, with `dataclasses.is_dataclass` as the predicate.                                                                                                                                                                                                                                                                                        |
| `iterator/registry.py`, `interface.py`        | Same, `DataclassChildFinder`. `enter_dataclasses` (opt-in) registers the fallback instead of the marker; its fate is decided by the `iterator` design and the `recursive` design (D1 there).                                                                                                                                                               |
| `recursive/registry.py`, `interface.py`       | Same, `DataclassTransformer`. the `recursive` design must read "register the fallback `dataclasses.is_dataclass`" where it says "register `DataclassTransformer` on `DataclassMarker`". `recursive/dataclass.py` and the guide mention the marker.                                                                                                         |
| `nested/conversion.py`                        | `registry.register(DataclassMarker, DefaultTransformer(), exist_ok=True)` becomes `registry.register_fallback(dataclasses.is_dataclass, DefaultTransformer(), exist_ok=True)`: a dataclass is a leaf for `to_jsonable`.                                                                                                                                    |
| `random/registry.py`                          | `RandomManagerRegistry(Registry[str, BaseRandomManager], BaseRandomManager)` is unchanged; it gains pickling and `exist_ok` as keyword-only (check the callers).                                                                                                                                                                                           |
| `utils/singleton.py`, `testing/registries.py` | `temporary_default_registries` uses `copy()` and `_replace_with_copy`; no change of code, the docs say what a copy contains (fallbacks).                                                                                                                                                                                                                   |
| `pyproject.toml` (import-linter)              | Delete `ignore_imports = ["coola.registry.base -> coola.equality"]`. `docs/docs/dev/architecture.md` ("Package Layering") loses the sentence about the exception.                                                                                                                                                                                          |
| Tests that read `registry._state`             | `tests/unit/recursive/test_registry.py`, `tests/unit/hashing/test_registry.py`: `_state` is still a `dict` of types to handlers, so they keep running, but they should use the public `registry[list]`, `list in registry`, `len(registry)`.                                                                                                               |

## Decisions for the maintainers

**D1. Replace `DataclassMarker` and the hashing fallbacks by one fallback mechanism?**
_Recommended: yes._ It is one idea implemented twice, the marker's identity test is a bug
, and a predicate also solves the virtual-subclass limit of the MRO lookup (a user can register
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
cost. The cache holds at most 1,024 classes and is emptied by any change, so a program that
creates classes dynamically keeps at most 1,024 of them alive (a few MB) until the next
registration, instead of the last reference. Alternative: no cache; simpler by about 12 lines, but
the dataclass fallback predicates run on every lookup.

**Decision: yes.**

**D4. Delete `equal()`?** _Recommended: yes. Answered: no, it is kept (rewritten as the alternative below, see 3.8)._ Nothing in `src` uses it, it is wrong for two default
registries, and it is the only reason `registry` depends on `equality`. A test that needs to
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

## Not covered

- **`issubclass` lookup.** Not added to `resolve`: it would send `str` to the handler of `Sequence`
  (the reason in the current docstring). The fallbacks are the opt-in way.
- **Entry-point plugins**: how plugins are listed or disabled is a
  question of the packages that load them, not of the registry.
- **Scoped overrides** (`temporary_default_registries`): in `coola.utils.singleton`; unchanged.
- **Handler protocol check at registration.** The registry stays generic; the domain registries
  could validate their handlers, which is a separate item.
- **A registry shared between processes.** Out of scope; a copy per process is the model.
- **The per-call caches of the engines** (`recursive`, `iterator`, `equality`): they keep their
  design; this review makes the registry lookup they call cheaper but does not change them. The
  hashing registry has no per-call cache and benefits most.
- **The thread-local cycle state of `HasherRegistry.hash`**: tracked separately
  (hashing), outside this design.

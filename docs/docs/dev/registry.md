# Architecture of `coola.registry`

:book: This page describes the design of the `coola.registry` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/registry.md#properties); the API is in the [reference](../refs/registry.md).

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

Two classes carry everything. A domain registry **is** a `TypeRegistry` (it inherits, it does not
wrap), so it has no `_state` wrapper and one display. The dataclass rule is a fallback predicate, not
a marker type.

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
  is about 50 and a registration is rare) and swaps the reference, which is atomic, so no method needs
  its own snapshot or invalidation logic.
- `register_many` builds one dict for the whole mapping and publishes it once: atomic by
  construction, including for readers.
- `_publish(state)` assigns `_state` and calls `_changed()`, a hook that `TypeRegistry` overrides to
  empty the cache. Three hooks exist for the subclass, all no-ops in `Registry`: `_validate_key`,
  `_load_for(key)`, `_load_all()`.
- `exist_ok` is keyword-only everywhere (`register`, `register_many`, `register_fallback`).

### Fork and pickle

`_reset_after_fork` recreates the (only) lock. `__getstate__` returns `__dict__` without
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

- The cache is a plain strong `dict`, bounded (1,024 classes) and emptied by any change (`register`,
  `unregister`, `clear`, `register_fallback`, a lazy load). The miss path is under the lock, as the
  registration is, so a stale value cannot be written after a registration has emptied the cache. A
  program that creates classes dynamically keeps at most 1,024 of them alive (a few MB) until the next
  registration. A weak dictionary would not keep them, but is the more expensive structure; no cache
  at all is simpler by about 12 lines, but the dataclass fallback predicates would run on every
  lookup.
- The fallbacks sit between the types and `object`: a registered type always wins, then the first
  predicate that accepts the type, then `object`. `object` is the **last** resort in every registry,
  so a handler registered for `object` never hides a fallback. `resolve(object)` checks the fallbacks
  too (no predicate of the repository accepts `object`). A predicate also solves the virtual-subclass
  limit of the MRO lookup (a user can register `lambda t: issubclass(t, Sequence)`).
- `resolve` is the only lookup name. The domain aliases `has_<x>` and `find_<x>` are public API of
  five packages and read well at the call site: they are one-line delegations.

### Fallbacks

`TypeRegistry._fallbacks` is a `dict[Callable[[type], bool], V]`, replaced on write like `_state`.

| Method                                                     | Behaviour                                                                                                                                     |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `register_fallback(predicate, handler, *, exist_ok=False)` | Appends the pair. A predicate already registered raises `AlreadyRegisteredError` (`exist_ok=True` replaces it, and its position is kept). |
| `unregister_fallback(predicate) -> handler`                | Removes the pair and returns the handler; `KeyNotRegisteredError` (a `KeyError`) if absent.                                                   |
| `fallbacks() -> ItemsView`                                 | The `(predicate, handler)` pairs in order, as a snapshot.                                                                                     |
| `clear()`                                                  | Removes the types, the fallbacks and the pending lazy groups. `copy()` and `repr` include the fallbacks.                                      |

The dataclass rule of each package is one registration in its `get_default_registry`:

```python
registry.register_fallback(dataclasses.is_dataclass, DataclassSummarizer())
# equality: the dataclass must use the __eq__ generated by @dataclass
registry.register_fallback(has_generated_dataclass_eq, DataclassEqualityTester())
# hashing
registry.register_fallback(dataclasses.is_dataclass, DataclassHasher())
registry.register_fallback(_has_coola_hash_method, ContentHasher())
```

The dataclass handler is therefore not a visible "type" entry of `items()` / `len()`.

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
  propagates, so the next call fails the same way.
- A lazy factory may call `register` on the same registry (the lock is reentrant).
- `repr` lists the pending groups by package name and loads nothing; `copy()` copies the tuple and
  loads nothing.
- `register(key, ...)`, `unregister(key)`, `has(key)`, `get(key)` and `[key]` call `_load_for(key)`,
  so `has(np.ndarray)` does not depend on whether numpy was already looked up. Registering over a
  lazy default therefore needs `exist_ok=True`; this is documented.

### Errors

```python
class KeyNotRegisteredError(KeyError):  # __str__ is not repr()'d
class TypeNotRegisteredError(KeyNotRegisteredError): ...
class AlreadyRegisteredError(ValueError): ...  # the value is not acceptable in the current state
```

The message builders (`Type '<class ...>' is not registered (MRO: ...). Registered types: ...`) are
module-level functions of `type.py`.

### What a domain registry looks like

```python
class HasherRegistry(TypeRegistry[Hasher[Any]]):
    def has_hasher(self, data_type: type) -> bool:
        return self.has(data_type)

    def find_hasher(self, data_type: type) -> Hasher[Any]:
        return self.resolve(data_type)

    def hash(self, data, length=64, ignore_unhashable=False): ...
```

The domain registry defines only the aliases and its entry point: `__init__`, `copy`, the fork reset,
the display, the fallbacks, `clear` and `equal` are inherited from the base classes.

### `equal`

`equal(other)` compares two registries without any feature package:

```python
def equal(self, other: object) -> bool:
    if type(other) is not type(self):
        return False
    self._load_all()
    other._load_all()
    return self._state == other._state and self._fallbacks == other._fallbacks
```

- The states are replaced, never mutated, so the comparison needs **no lock** and no import of
  `coola.equality` (so `registry` stays a core package with no exception in the import-linter
  contract).
- Handlers are compared with `==`: a handler without `__eq__` is equal to itself only. A `copy()` is
  therefore equal to its source (the handlers are shared). Two registries built separately with
  handlers of the same class are equal only if the handlers define `__eq__`.
- There is no `equal_nan` argument (it has no meaning for `==`).

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

Registering twice raises `AlreadyRegisteredError` (a `ValueError`) unless `exist_ok=True`, which is
keyword-only everywhere.

## Dependents

| Package / file                                | Use                                                                                                                                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `equality/tester/registry.py`, `interface.py` | `EqualityTesterRegistry(TypeRegistry[...])`; the default registry registers the dataclass tester as a fallback on `has_generated_dataclass_eq`. `find_equality_tester` is `resolve`. |
| `hashing/registry.py`, `interface.py`         | `HasherRegistry(TypeRegistry[...])`; the enum rule is `_find_enum_hasher`; `hash` calls the engine. The fallbacks are `dataclasses.is_dataclass` and the `__coola_hash__` predicate. |
| `summary`, `iterator`, `recursive`            | The same shape, with `dataclasses.is_dataclass` as the predicate of the dataclass fallback.                                                              |
| `jsonable`                                    | `TransformerRegistry` of `recursive`; a dataclass is entered through the fallback.                                                                       |
| `random/registry.py`                          | `RandomManagerRegistry(Registry[str, BaseRandomManager], BaseRandomManager)`: a registry keyed by name, not by type.                                      |
| `utils/singleton.py`, `testing/registries.py` | `temporary_default_registries` uses `copy()`; a copy contains the fallbacks.                                                                              |
| `pyproject.toml` (import-linter)              | `registry` is a core package: it imports no feature package.                                                                                              |

## Design choices and trade-offs

**One fallback mechanism instead of a marker type.** Before the fallbacks, the dataclass rule needed a
marker type in the registry and an identity test. A predicate is one idea implemented once, and it also
solves the virtual-subclass limit of the MRO lookup. The consequence is that the dataclass handler is
not a visible entry of `items()`. Keeping a marker type would leave each registry with a second
mechanism.

**A generic `Registry` is kept.** `RandomManagerRegistry` uses it and `TypeRegistry` extends it, so it
costs nothing: it is the storage class. Making it private would only change the guide (a "plugin
system" or "factory" example is not what `coola` is for).

**`equal()` is kept.** It is the documented way to compare two registries, it needs no feature
package, and the alternative (`dict(a.items()) == dict(b.items())`) forgets the fallbacks.

**`AlreadyRegisteredError` is a `ValueError`.** The value is not acceptable in the current state; it is
not a `RuntimeError`.

**Registries are picklable.** Ten lines remove a failure that only shows with `spawn`. The alternative
is to document that a registry is not picklable and to pass the arguments that build it.

**A duplicate fallback predicate is an error.** A silent duplicate would be appended and never reached.

## Not covered

- **`issubclass` lookup.** Not added to `resolve`: it would send `str` to the handler of `Sequence`
  (the reason in the current docstring). The fallbacks are the opt-in way.
- **Entry-point plugins**: how plugins are listed or disabled is a
  question of the packages that load them, not of the registry.
- **Scoped overrides** (`temporary_default_registries`): in `coola.utils.singleton`.
- **Handler protocol check at registration.** The registry stays generic; the domain registries
  could validate their handlers, which is a separate item.
- **A registry shared between processes.** Out of scope; a copy per process is the model.
- **The per-call caches of the engines** (`recursive`, `iterator`, `equality`, `hashing`): they
  belong to the engines; the registry cache makes the lookup they call cheap.

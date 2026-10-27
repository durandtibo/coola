# Registry System

:book: This page describes the `coola.registry` package, which provides thread-safe registry
implementations for storing and managing typed mappings.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.registry` package provides two types of registries for managing key-value mappings:

- **`Registry`**: A generic key-value registry for any hashable keys
- **`TypeRegistry`**: A specialized type-based registry with Method Resolution Order (MRO) support

Both registries are thread-safe and support standard dictionary operations through operator
overloading.

## Registry (Generic Key-Value)

### Basic Usage

The `Registry` class provides a thread-safe container for storing and retrieving values by key:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]()
>>> registry.register("key1", 42)
>>> registry.get("key1")
42

```

### Dictionary-Style Operations

The registry supports standard dictionary operations:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]()
>>> registry["key1"] = 100
>>> "key1" in registry
True
>>> registry["key1"]
100
>>> del registry["key1"]
>>> "key1" in registry
False

```

### Initialization with Data

You can initialize a registry with existing data:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]({"a": 1, "b": 2})
>>> len(registry)
2
>>> registry.get("a")
1

```

### Safe Registration

By default, registering a duplicate key raises `AlreadyRegisteredError` to prevent accidental
overwriting:

```pycon
>>> from coola.registry import AlreadyRegisteredError, Registry
>>> registry = Registry[str, int]()
>>> registry.register("key1", 42)
>>> try:
...     registry.register("key1", 100)
... except AlreadyRegisteredError:
...     print("already registered")
...
already registered

```

To allow overwriting, use `exist_ok=True`:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]()
>>> registry.register("key1", 42)
>>> registry.register("key1", 100, exist_ok=True)
>>> registry.get("key1")
100

```

### Bulk Registration

Register multiple key-value pairs at once:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]()
>>> registry.register_many({"key1": 42, "key2": 100, "key3": 7})
>>> len(registry)
3
>>> registry.get("key2")
100

```

Bulk updates with `exist_ok`:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]({"key1": 1})
>>> registry.register_many({"key1": 10, "key4": 4}, exist_ok=True)
>>> registry.get("key1")
10
>>> registry.get("key4")
4

```

### Unregistering Keys

Remove a key and get its value:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]({"key1": 42, "key2": 100})
>>> registry.has("key1")
True
>>> value = registry.unregister("key1")
>>> value
42
>>> registry.has("key1")
False

```

### Clearing Registry

Remove all entries from the registry:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]({"key1": 42, "key2": 100})
>>> len(registry)
2
>>> registry.clear()
>>> len(registry)
0

```

### Iterating Over Registry

Access keys, values, and items:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]({"a": 1, "b": 2})
>>> list(registry.keys())
['a', 'b']
>>> list(registry.values())
[1, 2]
>>> list(registry.items())
[('a', 1), ('b', 2)]

```

### Equality Checking

Compare two registries:

```pycon
>>> from coola.registry import Registry
>>> registry1 = Registry[str, int]({"key1": 42, "key2": 100})
>>> registry2 = Registry[str, int]({"key1": 42, "key2": 100})
>>> registry3 = Registry[str, int]({"key1": 42})
>>> registry1.equal(registry2)
True
>>> registry1.equal(registry3)
False

```

## TypeRegistry (Type-Based)

### Basic Usage

The `TypeRegistry` class provides a thread-safe container for mapping Python types to values:

```pycon
>>> from coola.registry import TypeRegistry
>>> registry = TypeRegistry[str]()
>>> registry.register(int, "I am an integer")
>>> registry.get(int)
'I am an integer'

```

### Type Resolution with MRO

The key feature of `TypeRegistry` is its `resolve()` method, which uses the Method Resolution Order
(MRO) to find the most appropriate value for a type:

```pycon
>>> from coola.registry import TypeRegistry
>>> registry = TypeRegistry[str]()
>>> registry.register(object, "I am an object")
>>> registry.register(int, "I am an integer")
>>> # Direct match
>>> registry.resolve(int)
'I am an integer'
>>> # Falls back to parent type via MRO (bool inherits from int)
>>> registry.resolve(bool)
'I am an integer'
>>> # Falls back to object
>>> registry.resolve(str)
'I am an object'

```

### Dictionary-Style Operations

Like `Registry`, `TypeRegistry` supports standard dictionary operations:

```pycon
>>> from coola.registry import TypeRegistry
>>> registry = TypeRegistry[str]()
>>> registry[str] = "I am a string"
>>> str in registry
True
>>> registry[str]
'I am a string'
>>> del registry[str]

```

### Initialization with Types

Initialize with type mappings:

```pycon
>>> from coola.registry import TypeRegistry
>>> registry = TypeRegistry[int]({str: 100, float: 200})
>>> len(registry)
2
>>> registry.get(str)
100

```

### Safe Type Registration

Prevent accidental overwriting of type mappings:

```pycon
>>> from coola.registry import AlreadyRegisteredError, TypeRegistry
>>> registry = TypeRegistry[int]()
>>> registry.register(str, 42)
>>> try:
...     registry.register(str, 100)
... except AlreadyRegisteredError:
...     print("already registered")
...
already registered

```

Allow overwriting with `exist_ok=True`:

```pycon
>>> from coola.registry import TypeRegistry
>>> registry = TypeRegistry[int]()
>>> registry.register(str, 42)
>>> registry.register(str, 100, exist_ok=True)
>>> registry.get(str)
100

```

### Bulk Type Registration

Register multiple types at once:

```pycon
>>> from coola.registry import TypeRegistry
>>> registry = TypeRegistry[str]()
>>> registry.register_many({int: "integer", float: "float", str: "string"})
>>> len(registry)
3
>>> registry.get(float)
'float'

```

### Inheritance-Based Lookup

The `resolve()` method walks the MRO to find the most specific registered type:

```pycon
>>> from coola.registry import TypeRegistry
>>> class Animal:
...     pass
>>> class Dog(Animal):
...     pass
>>> class Poodle(Dog):
...     pass
>>> registry = TypeRegistry[str]()
>>> registry.register(Animal, "animal")
>>> registry.register(Dog, "dog")
>>> registry.resolve(Dog)
'dog'
>>> registry.resolve(Poodle)  # Resolves to parent Dog
'dog'

```

!!! warning "Abstract base classes are only matched if they are real base classes"

    The lookup follows `__mro__`, it does not call `issubclass`. A registration for an abstract
    base class such as `collections.abc.Sequence` or `collections.abc.Mapping` therefore applies only
    to the classes that inherit from it. A class that is only _registered_ as a virtual subclass is
    not matched: `collections.deque` is a `MutableSequence` for `isinstance`, but it is not in the MRO
    of `Sequence`. Register such types explicitly (the default registries of `coola` register
    `deque` for this reason).

```pycon
>>> from collections import deque
>>> from collections.abc import Sequence
>>> from coola.registry import TypeRegistry
>>> registry = TypeRegistry[str]()
>>> registry.register(object, "object")
>>> registry.register(Sequence, "sequence")
>>> registry.resolve(list)  # list is not a subclass of Sequence
'object'
>>> registry.resolve(deque)
'object'
>>> registry.register(deque, "deque")
>>> registry.resolve(deque)
'deque'

```

### Performance Optimization

The `TypeRegistry` uses an internal cache for type resolution to optimize performance:

```pycon
>>> from coola.registry import TypeRegistry
>>> registry = TypeRegistry[str]()
>>> registry.register(object, "base")
>>> registry.register(int, "integer")
>>> # First resolve() populates cache
>>> registry.resolve(bool)
'integer'
>>> # Subsequent resolves use cached result
>>> registry.resolve(bool)
'integer'

```

The cache is automatically cleared when you register or unregister types to ensure consistency.

## Thread Safety

Both `Registry` and `TypeRegistry` are thread-safe. The state is a dictionary that is replaced on
every write and never mutated, so the reads take no lock, and `items()`, `keys()` and `values()`
return snapshots. The writers are serialized by a reentrant lock (`threading.RLock`). The lock is reentrant so that a lazy factory can register entries in the
registry that loads it:

```pycon
>>> from coola.registry import Registry
>>> import threading
>>> registry = Registry[str, int]()
>>> def worker(key, value):
...     registry.register(key, value, exist_ok=True)
>>> threads = [threading.Thread(target=worker, args=(f"key{i}", i)) for i in range(10)]
>>> for t in threads:
...     t.start()
>>> for t in threads:
...     t.join()
>>> len(registry) == 10
True

```

The registries are also safe to use in a process created with `os.fork` (a `multiprocessing` pool
with the `fork` start method, a `torch.utils.data.DataLoader` with workers). A lock held by a thread
of the parent at the time of the fork would stay locked forever in the child, so the child creates
new locks right after the fork and keeps the content of the registry. The same applies to
`LazySingleton`, to the default registries and to the lock of the savers. The `spawn` start method
is not affected.

## Lazy Registration

`register_lazy(names, factory)` registers a group of entries that is built the first time it is
needed. It is how `coola` registers the handlers of an optional library (numpy, torch, pandas, ...)
without importing the library. `names` is a package name or a sequence of package names, and
`factory` is a callable that returns a mapping, or its path `"package.module:name"`, which is
imported only when the group is loaded:

```pycon
>>> from coola.registry import TypeRegistry
>>> from decimal import Decimal
>>> registry = TypeRegistry[str]()
>>> registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
>>> registry.resolve(Decimal)
'decimal'

```

A group is loaded when a type from one of its packages is looked up, or when the whole content of
the registry is read (`len`, iteration, `items`, `keys`, `values`, `equal`). `repr` and `copy` never
load a group. The entries that are loaded never replace an entry registered by hand, and a group
whose factory raises stays pending, so the next access raises again. Because a lazy default is
visible to `register`, registering a type over it needs `exist_ok=True`.

## Copying a Registry

`copy()` creates an independent registry with the same entries (and the same pending lazy groups).
The keys and the values are not copied:

```pycon
>>> from coola.registry import Registry
>>> registry = Registry[str, int]({"key": 42})
>>> other = registry.copy()
>>> other.register("other", 1)
>>> registry.has("other")
False
>>> other.get("key")
42

```

Every registry of `coola` has a `copy()` method (`TypeRegistry`, `HasherRegistry`,
`EqualityTesterRegistry`, ...). To test code that registers a type in a default registry without
leaking the registration to the other tests, see "Isolating the Default Registries" in the
[testing guide](testing.md).

## Common Use Cases

### Plugin System

Use a registry to manage plugins:

```pycon
>>> from coola.registry import Registry
>>> plugin_registry = Registry[str, type]()
>>> class JSONParser:
...     pass
>>> class XMLParser:
...     pass
>>> plugin_registry.register("json", JSONParser)
>>> plugin_registry.register("xml", XMLParser)
>>> parser_class = plugin_registry.get("json")
>>> parser_class.__name__
'JSONParser'

```

### Type Dispatch

Use a type registry for dispatching based on type:

```pycon
>>> from coola.registry import TypeRegistry
>>> handlers = TypeRegistry[str]()
>>> handlers.register(int, "handle_int")
>>> handlers.register(str, "handle_str")
>>> handlers.register(list, "handle_list")
>>> handlers.resolve(int)
'handle_int'
>>> handlers.resolve(bool)  # bool inherits from int
'handle_int'

```

### Factory Pattern

Use a registry to implement a factory pattern:

```pycon
>>> from coola.registry import Registry
>>> factory = Registry[str, type]()
>>> class Circle:
...     pass
>>> class Square:
...     pass
>>> factory.register("circle", Circle)
>>> factory.register("square", Square)
>>> shape_class = factory.get("circle")
>>> shape_class.__name__
'Circle'

```

### Configuration Management

Store configuration handlers:

```pycon
>>> from coola.registry import Registry
>>> config_handlers = Registry[str, callable]()
>>> config_handlers.register("database", lambda: {"host": "localhost"})
>>> config_handlers.register("cache", lambda: {"enabled": True})
>>> db_config = config_handlers.get("database")()
>>> db_config
{'host': 'localhost'}

```

## Design Principles

The `coola.registry` package design provides:

1. **Thread safety**: The writers are serialized by a lock and the reads take none
2. **Type safety**: Generic typing support for type hints and IDE support
3. **Flexibility**: Support for both generic key-value and type-based registries
4. **Performance**: Caching in `TypeRegistry` for efficient type resolution
5. **Standard interface**: Dictionary-like operations for familiar API

## Properties

| Property                     | Statement                                                                                                                                                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Most specific type wins      | `TypeRegistry.resolve` returns the value of the first registered class of the Method Resolution Order, and falls back to `object` if it is registered.                                                          |
| No silent overwrite          | Registering an existing key raises `AlreadyRegisteredError`, unless `exist_ok=True`. `register_many` is atomic: nothing is registered if one key is invalid or a duplicate.                                      |
| Thread-safe, cheap reads     | A reader sees the state before or after a write, never a partial one, and a lookup takes no lock. The writers are serialized by a reentrant lock (`threading.RLock`).                                           |
| Stable views                 | `items()`, `keys()`, `values()` and iteration return snapshots: a later registration does not change a view already returned.                                                                                    |
| Independent copies           | `copy()` returns a registry of the same class with the same entries and pending lazy groups. Registering in one does not change the other. The values are shared, not copied.                                     |
| Fork-safe                    | A child created by `os.fork` creates new locks and keeps the content of the registry.                                                                                                                            |
| Lazy for optional libraries  | `register_lazy` imports nothing. A group is loaded when a type from its package is looked up or when the whole content is read. `repr` and `copy` never load. Loaded entries never replace explicit ones.       |
| Strict keys and clear errors | `TypeRegistry` keys are classes (`TypeError` otherwise). A missing key raises `KeyNotRegisteredError`, a missing type raises `TypeNotRegisteredError`. Both are `KeyError`.                                       |
| Insertion order              | `items()`, `keys()`, `values()` and iteration follow the order of registration.                                                                                                                                  |
| Cached resolution            | `TypeRegistry` caches the resolution of types for speed. Any change of the registry empties the cache.                                                                                                           |
| Generic typing               | Both registries are generic in their key and value types.                                                                                                                                                        |

## Comparison: Registry vs TypeRegistry

| Feature    | Registry                  | TypeRegistry              |
| ---------- | ------------------------- | ------------------------- |
| Key type   | Any hashable              | Python types only         |
| MRO lookup | No                        | Yes (via `resolve()`)     |
| Caching    | No                        | Yes (for type resolution) |
| Use case   | Generic key-value storage | Type-based dispatch       |

## See Also

- [`coola.recursive`](recursive.md): Its `TransformerRegistry` wraps a `TypeRegistry` internally
- [`coola.iterator`](iterator.md): Uses registry pattern for iterator dispatch
- [`coola.random`](random.md): Uses `Registry` to manage random managers

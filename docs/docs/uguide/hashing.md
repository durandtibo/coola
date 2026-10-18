# Hashing Objects

:book: This page describes the `coola.hashing` package, which provides deterministic hashing
of Python objects, including nested data structures.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.hashing` package computes a stable hash string for arbitrary objects, including
nested containers. Unlike Python's built-in `hash()`, which is randomized across processes for
strings by default and is not guaranteed to be stable across Python versions, `coola.hashing`
always returns the same hash for equivalent data. It offers:

1. **Deterministic hashes** - the same input always produces the same output
2. **Recursive hashing** - nested `list`, `dict`, `tuple`, and other containers are hashed
   element by element
3. **Type-specific hashers** - each Python type is handled by a hasher tailored to it
4. **Extensibility** - register hashers for custom types

## Basic Usage

The simplest way to hash an object is `hash_object`:

```pycon
>>> from coola.hashing import hash_object
>>> len(hash_object({"a": 1, "b": "abc"}))
64

```

`hash_object` uses the default global registry, which is pre-configured with hashers for common
built-in types (`str`, `bytes`, `int`, `float`, `bool`, `complex`, `list`, `tuple`, `dict`,
`set`, `date`, `datetime`, `Path`, ...). The same input always returns the same hash:

```pycon
>>> from coola.hashing import hash_object
>>> hash_object([1, 2, 3]) == hash_object([1, 2, 3])
True

```

### Controlling the Hash Length

The `length` argument controls the length of the returned hex string (an even number between 2
and 128, defaults to 64):

```pycon
>>> from coola.hashing import hash_object
>>> len(hash_object("hello", length=16))
16

```

### Handling Unhashable Types

By default, `hash_object` raises a `KeyError` if it encounters an object (including a nested
one) for which no hasher is registered:

```pycon
>>> from coola.hashing import hash_object
>>> try:
...     hash_object(object())
... except KeyError as e:
...     print("Error")
Error

```

Pass `ignore_unhashable=True` to replace unhashable objects with a deterministic placeholder
hash instead of raising an error. The placeholder only depends on the type, so any two
unsupported objects of the same type hash the same: a `UserWarning` is emitted each time a
placeholder is used.

```pycon
>>> import warnings
>>> from coola.hashing import hash_object
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     print(len(hash_object(object(), ignore_unhashable=True)))
64

```

Self-referencing containers (e.g. a list that contains itself) raise a `ValueError`.

## Supported Types

The default registry hashes **only** the following types (and their subclasses):

| Type                                         | Notes                                                                                             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `str`                                        | UTF-8 text, no Unicode normalisation                                                              |
| `bytes`, `bytearray`                         | A `bytes` and a `bytearray` with the same content hash differently                                |
| `int`, `float`, `complex`, `bool`, `None`    | The `repr` of the builtin value, so a NumPy scalar that is a `float` hashes like the same `float` |
| `Enum` (including `IntEnum`, `StrEnum`)      | The qualified class name and the member name, not the value                                       |
| `list`, `tuple`, `deque`, other `Sequence`   | Order matters. One tag per family: a `list` and a `tuple` with the same items hash differently    |
| `NamedTuple`                                 | Tagged with the class and field names, so it differs from a plain `tuple`                         |
| `dict`, other `Mapping`                      | Independent of the order of the keys                                                              |
| `OrderedDict`                                | Depends on the order of the keys, like its equality                                               |
| `set`, `frozenset`                           | Independent of the iteration order. A `set` and a `frozenset` hash differently                    |
| `date`, `datetime`, `time`                   | ISO 8601 form, including the UTC offset when there is one                                         |
| `timedelta`, `timezone`                      | The `repr` of the value                                                                           |
| `pathlib.PurePath`, `pathlib.Path`           | The POSIX form of the path as written, without touching the filesystem                            |
| pydantic `BaseModel` (if pydantic installed) | The qualified class name and the fields, hashed through the registry                              |

Nothing else is supported: **NumPy arrays, PyTorch tensors, pandas and polars data frames,
dataclasses and your own classes need a registered hasher** (use `DataclassHasher` for
dataclasses). Otherwise `hash_object` raises a
`TypeNotRegisteredError` (or hashes a placeholder with `ignore_unhashable=True`). There is no
default hasher for arrays and data frames because there is no canonical byte form for them: you
decide what must be part of the hash (the dtype? the memory layout? the index?). For example, for
NumPy arrays:

```pycon
>>> import numpy as np
>>> from coola.hashing import BaseHasher, hash_bytes, hash_object, register_hashers
>>> class NumpyHasher(BaseHasher[np.ndarray]):
...     def hash(self, data, registry, length=64, ignore_unhashable=False):
...         header = f"{data.dtype}|{data.shape}|".encode()
...         return hash_bytes(header + np.ascontiguousarray(data).tobytes(), length=length)
>>> register_hashers({np.ndarray: NumpyHasher()}, exist_ok=True)
>>> hash_object({"a": np.arange(3)}) == hash_object({"a": np.arange(3)})
True
>>> hash_object(np.arange(3)) == hash_object(np.arange(3).astype(float))
False

```

Values of different types do not collide: the type of the value is part of the hash, so `1`,
`1.0`, `True` and `"1"`, or `[]`, `()` and `""`, all have different hashes. Order matters for
ordered containers (`[1, 2]` differs from `[2, 1]`) and does not for unordered ones (`{1, 2}`
equals `{2, 1}`). The hash values can change between versions of
`coola`: do not persist them as a stable identifier across upgrades without checking.

## Type-Specific Hashers

Each hasher implements the `BaseHasher` interface and knows how to hash one kind of object,
delegating to a `HasherRegistry` to hash any nested values. The built-in hashers include:

- `StringHasher` / `hash_string` - hashes `str` objects, namespaced by the `str` type tag
- `BytesHasher` / `hash_bytes` - hashes `bytes` and `bytearray` objects, namespaced by the type tag
- `EnumHasher` / `hash_enum` - hashes `Enum` members by class name and member name
- `ReprHasher` - hashes an object's `repr()`; used for numeric types (`int`, `float`, `complex`,
  `bool`) because `repr()` preserves floating point precision
- `StrHasher` - hashes an object's `str()`; more human-readable but does not guarantee
  round-trip accuracy for floats
- `DatetimeHasher` - hashes `date`/`datetime`/`time` objects via their ISO 8601 representation
- `PathHasher` / `hash_path` - hashes `pathlib.Path` objects via their POSIX form as written, so
  hashes do not depend on the filesystem or the working directory
- `ResolvedPathHasher` - opt-in hasher that resolves the path first (symlinks, `..`, relative
  paths), so logically equivalent paths hash the same. The result depends on the filesystem and
  the current working directory
- `SequenceHasher` - hashes `list`, `tuple` and the other `Sequence` types element by element,
  under a configurable type tag
- `MappingHasher` - hashes `dict` and other `Mapping` types, sorted by key hash so that insertion
  order does not affect the result (unless `ordered=True`)
- `SetHasher` - hashes `set` and `frozenset`, sorted by item hash
- `DataclassHasher` - hashes dataclass instances from the class name and the field values
  (register it for each dataclass type)
- `PydanticModelHasher` / `hash_pydantic_model` - hashes pydantic `BaseModel` objects from the
  class name and the fields, with configurable handling of `SecretStr`/`SecretBytes` fields
- `HashableHasher` - hashes objects that implement the `SupportsHash` protocol (i.e. that define
  their own `hash()` method), namespaced by the class name

Each hasher can be used directly:

```pycon
>>> from coola.hashing import StrHasher, HasherRegistry
>>> registry = HasherRegistry()
>>> hasher = StrHasher()
>>> len(hasher.hash([1, 2, 3], registry=registry))
64

```

## The Hasher Registry

`HasherRegistry` maps Python types to hasher instances and dispatches to the most specific
registered hasher for an object's type, using the Method Resolution Order (MRO):

```pycon
>>> from coola.hashing import HasherRegistry, SequenceHasher, StrHasher
>>> from collections.abc import Sequence
>>> registry = HasherRegistry({object: StrHasher(), Sequence: SequenceHasher()})
>>> len(registry.hash([1, 2, 3]))
64

```

### The Default Registry

!!! tip "Unambiguous alias"

    `get_default_registry` exists in several `coola` subpackages with different return types.
    `get_default_hasher_registry` is an alias of the same function with a more explicit name, which avoids
    import collisions.

`get_default_registry` returns the singleton registry used internally by `hash_object`. Because
it is a singleton, any modification made via `register_hashers` affects every subsequent call:

```pycon
>>> from coola.hashing import register_hashers, StrHasher
>>> class Point:
...     def __init__(self, x, y):
...         self.x, self.y = x, y
...
...     def __str__(self):
...         return f"Point({self.x}, {self.y})"
>>> register_hashers({Point: StrHasher()})

```

`register_hashers` raises an `AlreadyRegisteredError` if a type is already registered, unless
`exist_ok=True` is passed to overwrite the existing registration.

If you need an isolated registry instead of mutating the shared singleton, create a
`HasherRegistry` instance directly, as shown above.

## Custom Objects

For a class that can compute its own hash, implement the `SupportsHash` protocol by defining a
`hash(registry=..., length=..., ignore_unhashable=...)` method. `HashableHasher` then dispatches
to it, and the registry-based hashers (`hash_object`, `SequenceHasher`, `MappingHasher`, ...) can
hash instances of that class like any other registered type.

## See Also

- [`coola.equality`](equality.md): For comparing nested data structures for equality
- [`coola.utils`](utils.md): For utility functions including import helpers

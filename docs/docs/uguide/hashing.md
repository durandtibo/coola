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
`date`, `datetime`, `Path`, ...). The same input always returns the same hash:

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
hash instead of raising an error:

```pycon
>>> from coola.hashing import hash_object
>>> len(hash_object(object(), ignore_unhashable=True))
64

```

## Type-Specific Hashers

Each hasher implements the `BaseHasher` interface and knows how to hash one kind of object,
delegating to a `HasherRegistry` to hash any nested values. The built-in hashers include:

- `StringHasher` / `hash_string` - hashes `str` objects directly
- `BytesHasher` / `hash_bytes` - hashes `bytes` objects directly
- `ReprHasher` - hashes an object's `repr()`; used for numeric types (`int`, `float`, `complex`,
  `bool`) because `repr()` preserves floating point precision
- `StrHasher` - hashes an object's `str()`; more human-readable but does not guarantee
  round-trip accuracy for floats
- `DatetimeHasher` - hashes `date`/`datetime` objects via their ISO 8601 representation
- `PathHasher` / `hash_path` - hashes `pathlib.Path` objects via their resolved POSIX form, so
  logically equivalent paths hash the same
- `SequenceHasher` - hashes `list`, `tuple`, `str`, and other `Sequence` types element by element
- `MappingHasher` - hashes `dict` and other `Mapping` types, sorted by key so that insertion
  order does not affect the result
- `PydanticModelHasher` / `hash_pydantic_model` - hashes pydantic `BaseModel` objects, with
  configurable handling of `SecretStr`/`SecretBytes` fields
- `HashableHasher` - hashes objects that implement the `SupportsHash` protocol (i.e. that define
  their own `hash()` method)

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

`register_hashers` raises a `RuntimeError` if a type is already registered, unless
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

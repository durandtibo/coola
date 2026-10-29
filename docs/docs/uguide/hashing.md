# Hashing Objects

:book: This page describes the `coola.hashing` package, which provides deterministic hashing
of Python objects, including nested data structures.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.hashing` package computes a stable hash string for arbitrary objects, including
nested containers. Unlike Python's built-in `hash()`, which is randomized across processes for
strings by default and is not guaranteed to be stable across Python versions, `coola.hashing`
always returns the same hash for equivalent data. The hashes are content fingerprints for
identity, caching and deduplication: they are **not** cryptographic guarantees, so do not use them
for passwords, signatures or integrity against an attacker. It offers:

1. **Deterministic hashes** - the same input always produces the same output
2. **Recursive hashing** - nested `list`, `dict`, `tuple`, and other containers are hashed
   element by element
3. **Type-specific hashers** - each Python type is handled by a hasher tailored to it
4. **Extensibility** - register hashers for custom types

## Properties

The default registry is designed around the following properties.

| Property                 | Statement                                                                                                                                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Equality-consistent      | Objects that are equal for [`objects_are_equal`](equality.md) have the same hash. The reverse can fail for a few types that share a hash on purpose, e.g. a `list` subclass and a `list`, or a NumPy `float64` and a `float`. |
| Deterministic            | The same value gives the same hash across calls, processes, machines and operating systems. It does not depend on `PYTHONHASHSEED`, on object addresses or on the iteration order of a set.                                   |
| Type-sensitive           | Values of different types hash differently, even when their text looks the same: `1`, `1.0`, `True`, `"1"`, `b"1"`, `memoryview(b"1")` and `[1]` are all distinct.                                                            |
| Structure-sensitive      | Nesting and item boundaries matter: `[[1], 2]`, `[1, [2]]` and `[1, 2]` are distinct, and so are `["ab", "c"]` and `["a", "bc"]`.                                                                                             |
| Order rules              | Ordered containers are order-sensitive (`[1, 2]` differs from `[2, 1]`). Unordered containers are not (`{1, 2}` equals `{2, 1}`). `OrderedDict` is ordered, like in `objects_are_equal`.                                      |
| Composable               | A container hash is built only from the digests of its children, resolved through the registry, so each type can be extended independently.                                                                                   |
| Fixed width, user-chosen | The output is a lowercase hex string of `length` characters (an even number between 2 and 128, default 64). It is the first `length` characters of the full 128-character hash.                                               |
| No silent false equality | An unsupported type fails loudly by default. The `ignore_unhashable` placeholder is an explicit, documented opt-out that warns once per type.                                                                                 |
| Cycle-safe               | A container that contains itself raises `ValueError`, whichever hasher is registered for it.                                                                                                                                  |
| Any depth                | No `RecursionError`, whatever the nesting or the caller's stack: the engine uses an explicit stack. The depth is bounded by memory only.                                                                                      |
| Linear cost              | The cost is linear in the number of **objects**, not of paths: a container reachable by several paths (a DAG) is hashed once per call.                                                                                        |
| Pure, thread-safe        | Hashing has no side effect and keeps no global state (no thread-local, no lock), so a registry and a hasher can be shared by threads. Reading the filesystem is an explicit opt-in (`ResolvedPathHasher`).                    |
| Errors propagate         | An exception of a hasher propagates unchanged. An invalid `length` raises `ValueError` before any traversal. A secret is never hashed unless asked for, wherever it is.                                                       |
| Cross-library scalars    | Subclasses of `bool`, `int`, `float` and `complex` (such as `np.float64`) hash like the equivalent builtin, independently of the library version.                                                                             |
| Consistent with siblings | Dataclasses and pydantic models are entered, only their `compare=True` fields count (like `objects_are_equal`); options are keyword-only; one extension point (`encode`).                                                     |

Not guaranteed:

- **Stability across versions of `coola`.** The format can change in a major release (the format
  version is part of every digest, so two formats are never confused). `get_stable_registry()`
  makes the stable identifiers independent of the installed plugins.
- **Stability across versions of a library** when the library changes what an object means (the
  dtype names of pandas, for example).
- **Hashes of classes that are renamed or moved.** A dataclass, a model, an enum and a custom
  object hash with their qualified class name.
- **Cycles.** A cyclic structure raises `ValueError`, although `objects_are_equal` accepts it.
- **Resistance to an adversary.** Do not use a hash as a security boundary.

### The encoding

Every hash is derived from one rule. A hasher describes a value as a leaf (`tag`, a NUL byte and a
payload) or as a node (a tag and the children to hash). The engine computes

```
digest(x)  = BLAKE2b-512(preimage, person = b"coola-hash-2")          # 64 bytes
leaf       : preimage = 0x00 + tag + 0x00 + payload
node       : preimage = 0x01 + tag + 0x00 + e1 + e2 + ... + en         # one entry per child
entry      : 0xFF + 64-byte digest of the child, or, if the child is a leaf of at most 64 bytes,
             its size (one byte) + the leaf itself ("inlined leaf")
unordered  : the entries (a key and its value) are sorted
hash_object(x, length=n) = digest(x).hex()[:n]
```

The entries are self-delimiting (a marker `0xFF` and a fixed width, or a size), which makes the
concatenation unambiguous without other delimiters. A short leaf (a number, a short string) is part
of the preimage of its parent instead of a BLAKE2b call of its own, which is what makes the flat
containers fast. The leading `0x00`/`0x01` byte makes a leaf and a node with the same tag different. The
registry dispatches on `type(data)` along the Method Resolution Order, so the most specific
registered type wins. A scalar and a standard-library container have a constant tag, so a subclass
hashes like its base; a user-defined class (dataclass, model, enum, custom object) uses its
qualified name.

### Choices that may surprise you

| Question                                           | Answer                                                                                                                                                    |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[1, 2]` vs `(1, 2)`                               | **Different.** They are different types, and each sequence family has its own tag.                                                                        |
| `list` vs `deque`                                  | **Different**, for the same reason.                                                                                                                       |
| `1` vs `1.0` vs `True`                             | **All different.** `1 == 1.0` in Python, but the identity of a hash is the type and the value.                                                            |
| `{"a": 1, "b": 2}` vs `{"b": 2, "a": 1}`           | **Same.** Insertion order is not semantic for mappings.                                                                                                   |
| `dict` vs `OrderedDict`                            | **Different**, like in `objects_are_equal`.                                                                                                               |
| `OrderedDict` with the same items in another order | **Different.** Order is the point of an `OrderedDict`, and `objects_are_equal` also says they are different.                                              |
| `set` vs `frozenset`                               | **Different.** Both are order-insensitive.                                                                                                                |
| `set` vs `list` with the same items                | **Different.**                                                                                                                                            |
| `str` vs `bytes`                                   | **Different.**                                                                                                                                            |
| `IntEnum` / `StrEnum` member vs its value          | **Different.** Enum members are not collapsed into `int` or `str`.                                                                                        |
| Subclass instance vs base instance                 | **Same** for scalars normalised to the builtin (e.g. NumPy floats), and for subclasses that dispatch to the same hasher and tag (e.g. a `list` subclass). |

```pycon
>>> from coola.hashing import hash_object
>>> hash_object([1, 2]) != hash_object((1, 2))
True
>>> hash_object(1) != hash_object(1.0)
True
>>> hash_object({"a": 1, "b": 2}) == hash_object({"b": 2, "a": 1})
True
>>> hash_object({1, 2, 3}) == hash_object({3, 2, 1})
True

```

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

The options (`registry`, `length` and `ignore_unhashable`) are keyword-only, because
`hash_object(data, 32)` would pass `32` as the registry.

### Controlling the Hash Length

The `length` argument controls the length of the returned hex string (an even number between 2
and 128, defaults to 64). A `bool` or a `float` is rejected like an odd number. Hashes of different
lengths are prefixes of one another: `hash_object(x, length=16)` is the first 16 characters of
`hash_object(x, length=128)`. A short identifier can therefore be matched against a stored longer
hash with `startswith`, and changing the length later does not change the start of the hashes you
already have.

```pycon
>>> from coola.hashing import hash_object
>>> len(hash_object("hello", length=16))
16
>>> hash_object([1, 2], length=128).startswith(hash_object([1, 2], length=16))
True

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
hash instead of raising an error. The placeholder only depends on the type, so it is **lossy**:
any two unsupported objects of the same type hash the same. A `UserWarning` is emitted once per type
in each call, so a silent false equality does not hide in a cache key.

```pycon
>>> import warnings
>>> from coola.hashing import hash_object
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore")
...     print(len(hash_object(object(), ignore_unhashable=True)))
64

```

Self-referencing containers (e.g. a list that contains itself) raise a `ValueError`. There is no
limit on the nesting depth and no `RecursionError`, and a container that is reachable by several
paths is hashed once, so a structure with `2**60` paths takes a linear time:

```pycon
>>> from coola.hashing import hash_object
>>> data = []
>>> for _ in range(60):
...     data = [data, data]
>>> len(hash_object(data))
64

```

## Supported Types

The default registry hashes **only** the following types (and their subclasses):

| Type                                         | Notes                                                                                                                |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `str`                                        | UTF-8 text, no Unicode normalisation                                                                                 |
| `bytes`, `bytearray`                         | A `bytes` and a `bytearray` with the same content hash differently                                                   |
| `int`, `float`, `complex`, `bool`, `None`    | The `repr` of the builtin value, so a NumPy scalar that is a `float` hashes like the same `float`                    |
| `Enum` (including `IntEnum`, `StrEnum`)      | The qualified class name and the member name, not the value                                                          |
| `list`, `tuple`, `deque`, other `Sequence`   | Order matters. One tag per family: a `list` and a `tuple` with the same items hash differently                       |
| `NamedTuple`                                 | Tagged with the class and field names, so it differs from a plain `tuple`                                            |
| `dict`, other `Mapping`                      | Independent of the order of the keys                                                                                 |
| `OrderedDict`                                | Own tag. The order of the keys matters, like in `objects_are_equal`                                                  |
| `set`, `frozenset`                           | Independent of the iteration order. A `set` and a `frozenset` hash differently                                       |
| `date`, `datetime`, `time`                   | ISO 8601 form; aware `datetime` and `time` are normalised to UTC                                                     |
| `timedelta`, `timezone`                      | The `repr` of the value                                                                                              |
| `uuid.UUID`, `zoneinfo.ZoneInfo`             | The string form (the UUID text, the zone key)                                                                        |
| `decimal.Decimal`                            | The normalised value: `Decimal('1.0')` and `Decimal('1.00')` have the same hash                                      |
| `fractions.Fraction`                         | The numerator and the denominator in lowest terms                                                                    |
| `range`                                      | The items, not the arguments: `range(0)` and `range(5, 5)` have the same hash, in constant time                      |
| NumPy scalars (`np.float32`, `np.int64`...)  | Like the equivalent builtin value: `np.float32(1.5)` hashes like `1.5`                                               |
| dataclass instances                          | The qualified class name and the fields, hashed through the registry                                                 |
| objects with a `__coola_hash__()` method     | The value of their `__coola_hash__()` (see `SupportsCoolaHash`), tagged with the class name                          |
| `pathlib.PurePath`, `pathlib.Path`           | The POSIX form of the path as written, without touching the filesystem                                               |
| pydantic `BaseModel` (if pydantic installed) | The qualified class name and the fields, hashed through the registry                                                 |
| `numpy.ndarray`, `torch.Tensor`, `jax.Array` | The dtype, the shape and the values (not the memory layout, the device or `requires_grad`); `-0.0` hashes like `0.0` |
| `pandas`, `polars` data frames and series    | The names, the dtypes, the index (pandas) and the values                                                             |
| `pyarrow` arrays and tables                  | The names, the types and the values; the chunk layout is ignored                                                     |
| `xarray` `DataArray`, `Dataset`, `Variable`  | The dimensions, the values, the coordinates, the attributes and the name                                             |

Some types that `objects_are_equal` compares are not hashed yet: `array.array`, `dict_keys`,
`dict_values`, `dict_items`, `slice`, `Ellipsis`, `re.Pattern`, `torch.dtype`, `torch.device`, the
NumPy `datetime64` and `timedelta64` scalars, and the pandas `Index` and `Categorical` as values.
Functions, classes and other callables are not hashed either, and the content of a file is not
hashed (see `Path` below). Each of them is one hasher and one registration if you need it.

The array libraries are only imported if they are installed. A NumPy array and a PyTorch tensor with
the same content hash differently. Your own classes still need a registered hasher (a class with a
`__coola_hash__()` method and a dataclass work without registration, see below). Otherwise `hash_object`
raises a `TypeNotRegisteredError` that lists the remedies (or hashes a placeholder with
`ignore_unhashable=True`). To change what is part of the hash, register your own hasher, for
example for NumPy arrays:

```pycon
>>> import numpy as np
>>> from coola.hashing import BaseHasher, digest, hash_object, register_hashers, tagged
>>> class NumpyHasher(BaseHasher[np.ndarray]):
...     def encode(self, data):
...         header = f"{data.dtype}|{data.shape}|".encode()
...         return tagged("array", header + digest(np.ascontiguousarray(data).tobytes()))
>>> register_hashers({np.ndarray: NumpyHasher()}, exist_ok=True)
>>> hash_object({"a": np.arange(3)}) == hash_object({"a": np.arange(3)})
True
>>> hash_object(np.arange(3)) == hash_object(np.arange(3).astype(float))
False

```

The hash values can change between versions of `coola`: do not persist them as a stable
identifier across upgrades without checking. The unit tests keep one golden vector per hasher
family (`tests/unit/hashing/test_golden.py`), so a change of the format is a deliberate edit.

`hash_object` uses the default registry, which changes when code calls `register_hashers` or when
an installed package declares a `coola.hashing.hashers` entry point. The identifiers
`generate_stable_content_id` and `generate_stable_uuid5` use `get_stable_registry()` instead (the
built-in hashers only) when no registry is given, so they do not depend on the installed packages.
Passing another registry opts out of this stability.

## Behavior per Type

- **`str`**: tag `str`, UTF-8 payload. There is no Unicode normalisation, so a precomposed `é` and
  `e` followed by a combining accent are different strings. A lone surrogate (for example in a
  file name from `os.fsdecode`) is encoded with `surrogatepass`, which never collides with a valid
  string.
- **`bytes`, `bytearray`**: tagged with their type name, payload is the raw content.
- **`bool`, `int`, `float`, `complex`, `None`**: tag is the type name, payload is the builtin
  `repr`, which round-trips exactly. `bool` is checked before `int`, so `True` and `1` differ.
  `-0.0` hashes like `0.0` (they are equal), and all NaNs hash the same.
- **`Enum`**: the qualified class name and the member name, not the member value.
- **`list`, `tuple`, `deque`, other `Sequence`**: the tag is the sequence family, the payload is
  the child hashes in order. A named tuple also includes its class name and field names, so
  `Point(1, 2)` differs from `(1, 2)`. Strings and bytes are never iterated as sequences.
- **`dict`, other `Mapping`**: each item is `key_hash + value_hash`, and items are sorted by the
  _key hash_, not by the key. Keys do not need to be orderable and can have mixed types.
  `OrderedDict` is the exception: its items are hashed in order, so two `OrderedDict`s with the same
  items in a different order hash differently, like `objects_are_equal` says they are different.
  Register `MappingHasher(tag, ordered=True)` for another mapping type whose order is meaningful.
- **`set`, `frozenset`**: the sorted item hashes under the `set` or `frozenset` tag.
- **`date`, `datetime`, `time`**: tagged with the concrete class name, payload is the ISO 8601
  string. Aware datetimes are converted to UTC first, so the same instant in two time zones hashes
  the same, while a naive and an aware datetime differ. An aware `time` is normalised the same way
  (the offset is subtracted from the time of day), so equal aware times hash the same. `timedelta`
  and `timezone` use their `repr`.
- **`Path`, `PurePath`**: the POSIX form of the path as written, with no filesystem access, so the
  hash is portable across machines. A relative and an absolute path to the same location differ.
  Register `ResolvedPathHasher` to resolve symlinks, `..` and relative paths first; the result
  then depends on the filesystem and on the current working directory. Hashing the _content_ of
  a file is a different operation that is not provided by a hasher.
- **pydantic `BaseModel`**: the qualified model class name plus the fields, hashed as a mapping
  through the registry (extra fields included). Two different model classes with the same values
  therefore differ. A secret (`SecretStr`, `SecretBytes`) is a value with a hasher, so the policy is
  the one of the registry and applies wherever the secret is (in a model, a `dict`, a dataclass...).
  The default registry raises a `ValueError` (the message never contains the secret). To change the
  policy, build a registry with `get_secret_hashers`:
  - `"error"` (default) raises a `ValueError`,
  - `"reveal"` hashes the real secret value,
  - `"exclude"` hashes a constant, so objects that differ only by a secret hash the same (the
    position of the secret is kept).

  ```pycon
  >>> from pydantic import BaseModel, SecretStr
  >>> from coola.hashing import get_default_registry, get_secret_hashers
  >>> class Creds(BaseModel):
  ...     user: str
  ...     password: SecretStr
  >>> registry = get_default_registry().copy()
  >>> registry.register_many(get_secret_hashers("exclude"), exist_ok=True)
  >>> registry.hash(Creds(user="a", password="x")) == registry.hash(Creds(user="a", password="y"))
  True

  ```

- **Dataclasses**: the qualified class name plus the field values. The default registry hashes every
  dataclass with `DataclassHasher` (a fallback, see [Fallbacks](#fallbacks)), without registration.
- **Objects with their own `hash` method**: the result of `hash`, tagged with the qualified class
  name (see [Custom Objects](#custom-objects)).

```pycon
>>> from dataclasses import dataclass
>>> from coola.hashing import hash_object
>>> @dataclass
... class Point:
...     x: int
...     y: int
>>> hash_object(Point(1, 2)) == hash_object(Point(1, 2))
True
>>> hash_object(Point(1, 2)) != hash_object({"x": 1, "y": 2})
True

```

## Errors

| Situation                           | Behavior                                                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `length` is odd or outside 2 to 128 | `ValueError`                                                                                                  |
| No hasher for the type              | `TypeNotRegisteredError` (a `KeyError`) naming the type, or a lossy placeholder with `ignore_unhashable=True` |
| Self-referencing container          | `ValueError("Cannot hash a self-referencing <type>.")`                                                        |
| Deep nesting                        | Works: the only limit is the memory                                                                           |
| An exception of a hasher            | Propagates unchanged                                                                                          |
| Secret with the default registry    | `ValueError`, without leaking the value                                                                       |

## Type-Specific Hashers

Each hasher implements the `BaseHasher` interface and describes one kind of object with `encode`;
the engine hashes the children of a `Node` with the hasher of their own type. The built-in hashers include:

- `StringHasher` - hashes `str` objects, namespaced by the `str` type tag
- `BytesHasher` - hashes `bytes` and `bytearray` objects, namespaced by the type tag
- `EnumHasher` - hashes `Enum` members by class name and member name
- `ReprHasher` - hashes an object's `repr()`; used for numeric types (`int`, `float`, `complex`,
  `bool`) because `repr()` preserves floating point precision
- `StrHasher` - hashes an object's `str()`; more human-readable but does not guarantee
  round-trip accuracy for floats
- `DatetimeHasher` - hashes `date`/`datetime`/`time` objects via their ISO 8601 representation
- `PathHasher` - hashes `pathlib.Path` objects via their POSIX form as written, so
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
  (used for every dataclass by the default registry)
- `PydanticModelHasher` - hashes pydantic `BaseModel` objects from the class name and the fields
- `SecretHasher` / `get_secret_hashers` - the policy (`"error"`, `"reveal"`, `"exclude"`) for
  `SecretStr` and `SecretBytes`, wherever they are
- `ContentHasher` - hashes objects that implement the `SupportsCoolaHash` protocol (i.e. that
  define a `__coola_hash__()` method that returns their content), namespaced by the class name
- `NumpyArrayHasher`, `TorchTensorHasher`, `JaxArrayHasher` - hash the dtype, the shape and the
  values of an array or of a tensor (require the corresponding package)
- `PandasSeriesHasher`, `PandasDataFrameHasher`, `PolarsSeriesHasher`, `PolarsDataFrameHasher` - hash
  the names, the dtypes and the values of a series or of a data frame (require the corresponding
  package)
- `PyarrowArrayHasher`, `PyarrowTableHasher` - hash the types and the values of a PyArrow array or
  table (require `pyarrow`)
- `XarrayHasher` - hashes the dimensions, the values, the coordinates, the attributes and the name of
  an xarray `DataArray`, `Dataset` or `Variable` (requires `xarray`)

A hasher describes the content of a type with `encode`; the engine does the hashing. Each hasher
can be used directly to see that description:

```pycon
>>> from coola.hashing import SequenceHasher, StrHasher
>>> StrHasher().encode(1)
b'str:int\x001'
>>> SequenceHasher("list").encode([1, 2, 3])
Node(tag='list', children=[1, 2, 3], ordered=True, group=1)

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

### Fallbacks

A registry maps _types_ to hashers, so it cannot say "every dataclass" or "every class with a
`__coola_hash__()` method". `HasherRegistry.register_fallback(predicate, hasher)` registers a hasher for the
types that satisfy a predicate. A fallback is only used if no hasher is registered for the type or
for one of its parent types (`object` excluded), so a hasher registered for a type always wins. The default registry
has two fallbacks: dataclasses (`DataclassHasher`) and the classes with a `__coola_hash__()` method
(`ContentHasher`). A dataclass that also defines `__coola_hash__()` uses the dataclass hasher.

```pycon
>>> from coola.hashing import HasherRegistry, SequenceHasher, DataclassHasher
>>> from dataclasses import dataclass, is_dataclass
>>> registry = HasherRegistry({list: SequenceHasher()})
>>> registry.register_fallback(is_dataclass, DataclassHasher())
>>> @dataclass
... class Point:
...     x: int
>>> registry.find_hasher(Point)
DataclassHasher()

```

The hasher registered for `object` is the last resort: the fallbacks are tried before it. A
predicate can only be registered once (`AlreadyRegisteredError`, unless `exist_ok=True`, which keeps
its position); if several predicates match, the first registered one wins.

`HasherRegistry.unregister_fallback(predicate)` removes a fallback and returns its hasher, `clear()` removes the hashers
and the fallbacks, and `equal()` compares the fallbacks (predicates, in order, and hashers) as well
as the hashers.

## Custom Objects

For a class that cannot be a dataclass, define a `__coola_hash__()` method that returns the content
of the object (any value that can be hashed). `ContentHasher` hashes that value through the
registry, with the qualified class name as tag, so two classes with the same content hash
differently and the object cannot bypass the protections of the engine (cycles, depth,
`length`):

```pycon
>>> from coola.hashing import hash_object
>>> class Point:
...     def __init__(self, x, y):
...         self.x, self.y = x, y
...
...     def __coola_hash__(self):
...         return (self.x, self.y)
>>> hash_object(Point(1, 2)) == hash_object(Point(1, 2))
True
>>> hash_object(Point(1, 2)) == hash_object((1, 2))
False

```

### Writing a Hasher

A hasher only describes the content of a type: `encode(data)` returns the bytes of a leaf
(`tagged(tag, payload)`) or a `Node(tag, children)`. The engine hashes the children with the
hasher of their own type, handles the cycles, the shared containers and the depth, and applies
`length` and `ignore_unhashable`. When you add a hasher, follow these rules so that it keeps the
guarantees above:

1. Pick a unique, stable tag, and never reuse the tag of another type family. The tag is a
   constant of the hasher, except for user-defined classes, whose qualified name is the identity.
2. Return `tagged(...)` for a leaf, or a `Node` for a container. Reduce a big buffer with
   `digest(buffer)` before it enters `tagged`.
3. Do not call the registry and do not recurse: put the children in the `Node`. Use
   `ordered=False` (and `group=2` for a key and a value) when the order is not meaningful.
4. Do not depend on object addresses, `id()`, `hash()`, the iteration order of dicts or sets,
   the locale, the time zone or the clock.
5. Register the hasher for the most general type whose semantics are identical, and for the
   exact type where they are not.
6. Test stability, equality-consistency, type-sensitivity, item boundaries and order.

## See Also

- [`coola.equality`](equality.md): For comparing nested data structures for equality
- [`coola.utils`](utils.md): For utility functions including import helpers

# Architecture of `coola.hashing`

:book: This page describes the design of the `coola.hashing` package for contributors: the invariants, the contracts between the parts, and the trade-offs of the main design choices. The rules that users rely on are in the [user guide](../uguide/hashing.md#properties); the API is in the [reference](../refs/hashing.md).

## Overview

The package has **one engine, one extension point and one encoding rule**. A hasher only describes
the content of a type: `encode(data)` returns the bytes of a leaf (`tagged(tag, payload)`) or a
`Node(tag, children, ordered, group)`. The engine (`coola.hashing.engine`) owns everything else: the
dispatch by type (once per type and per call), the digests, the order rule, the cycles, the shared
containers, the depth and the placeholder of `ignore_unhashable`. This is the same split as
`equality` (the engine owns identity, type, depth and cycles; a tester returns the child pairs) and
`recursive` (a transformer returns the children).

```
hash_object(data, *, registry=None, length=64, ignore_unhashable=False)
        |   validates length; one registry; one call = one engine run
        v
digest_of(data, registry, ...)       explicit stack of frames; per-call type -> hasher cache and memo
        |-- hasher   = registry.find_hasher(type(data))     MRO, fallbacks, Enum rule
        |-- encoding = hasher.encode(data)                  pure: no registry, no recursion
        |        bytes -> leaf:  digest = BLAKE2b(0x00 + encoding)
        |        Node  -> push the children; when done:
        |                 digest = BLAKE2b(0x01 + tag + 0x00 + entries of the children)
        |                 (an entry is 0xFF + digest, or size + leaf for a leaf of at most 64 bytes)
        v
the 64-byte digest of the root  ->  hex()[:length]
```

| Design choice                                                       | Why                                                                      |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `encode` returns a description, the engine hashes                   | cycles, depth, sharing, `length` and the placeholder cannot be forgotten |
| A leaf of at most 64 bytes is inlined in the preimage of its parent | a flat container makes one BLAKE2b call instead of one per item          |
| One 64-byte digest, cut at the end by `length`                      | a hasher never sees `length`; a short hash is the start of a longer one  |
| The format version is the BLAKE2b `person` parameter                | two formats can never be confused; a change is one constant              |
| A shared container is hashed once per call (a memo by `id`)         | a hash is pure, so the memo is invisible; a DAG is linear                |
| A cycle raises `ValueError`                                         | the digest of a container in a cycle would depend on where it is reached |
| Secrets are types with a hasher (`SecretHasher`)                    | the policy is the one of the registry, wherever the secret is            |
| `__coola_hash__` returns the content of a custom object, not a hash | the object cannot bypass the engine and does not forward any option      |
| Only the `compare=True` fields of a dataclass                       | equal dataclasses have the same hash                                     |

See the [user guide](../uguide/hashing.md#properties) for the properties of the package. The
trade-offs behind these choices are in [Design choices and trade-offs](#design-choices-and-trade-offs).

## Design constraints

1. **Simple and easy to maintain.** A hasher only describes a type. It does not repeat the
   `registry`, `length` and `ignore_unhashable` parameters, and it carries no cycle state, no
   `RecursionError` translation and no signature inspection: the engine owns them.
2. **Stable, versioned format.** The encoding is documented, and the digests are covered by golden
   vectors, so a change of any hash value is deliberate. Every stable identifier is built on
   `hash_object`, so it changes with the format; the format version (the BLAKE2b `person` parameter)
   makes two formats impossible to confuse.
3. **Consistent with equality.** "Equality drives hashing, not the reverse": equal objects must have
   the same hash, and where a hasher disagrees with `objects_are_equal` the hasher is changed.
4. **No slower than the benchmark gate.** `tests/benchmarks/test_hashing_benchmark.py` measures the
   engine. A design that is simpler but slower is rejected.
5. **Consistent with the siblings.** The same input is handled the same way by `equality`,
   `summary`, `recursive`, `iterator` and `hashing`: dataclasses are entered, cycles are detected,
   there is no `RecursionError`, options are keyword-only, the engine owns the invariants and the
   type-specific class only describes its type.
6. **Optional libraries stay lazy.** The package imports no optional library; the hashers of
   NumPy, torch, JAX, pandas, polars, pyarrow, xarray and pydantic are loaded the first time the
   registry looks for one of their types (`register_lazy`).

## Architecture in detail

```
hash_object(data, *, registry=None, length=64, ignore_unhashable=False)
        |   validates length; one registry (default: the singleton); one call = one engine run
        v
digest_of(data, registry, ignore_unhashable)           coola/hashing/engine.py
        |   explicit stack of frames; per-call type -> hasher cache; per-call memo / ancestors
        |
        |-- hasher   = registry.find_hasher(type(data))        MRO, fallbacks, Enum rule
        |-- encoding = hasher.encode(data)                     pure: no registry, no recursion
        |        bytes  -> leaf: digest = BLAKE2b(0x00 + encoding)
        |        Node   -> push the children; when they are done:
        |                  digest = BLAKE2b(0x01 + tag + 0x00 + entries of the children)
        v
the 64-byte digest of the root  ->  hex()[:length]
```

The engine owns the invariants (dispatch cache, composition, order rule, cycles, depth, sharing,
placeholder). A hasher **only describes the content of a type**: a tagged byte string for a leaf, or
a tag and the children for a container. That is the same split as `equality` (the engine owns
identity, type, depth and cycles; a tester returns the child pairs) and `recursive` (a transformer
returns the children and how to rebuild). `registry`, `length` and `ignore_unhashable` are not
arguments of a hasher, so a hasher cannot forget a protection, cannot recurse and cannot leak a
parameter.

### The hasher contract

```python
@dataclass(slots=True)
class Node:
    tag: str  # constant of the hasher
    children: Iterable[Any]  # the values to hash, consumed once, lazily
    ordered: bool = True  # False: the groups of children are sorted by digest
    group: int = 1  # number of consecutive children that form one unordered entry


Encoding = bytes | Node


class Hasher(Protocol[T_contra]):
    def encode(self, data: T_contra) -> Encoding:
        """A leaf (``tagged(tag, payload)``) or a ``Node`` that describes the content of ``data``."""


class BaseHasher(ABC, Generic[T]):  # convenience base class
    @abstractmethod
    def encode(self, data: T) -> Encoding: ...
```

Two helpers are public, so that a hasher never imports `hashlib` and the format stays in one place:

- `tagged(tag: str, payload: str | bytes = b"") -> bytes`: `tag`, a NUL byte, `payload` (text is
  encoded in UTF-8 with `surrogatepass`: a lone surrogate never collides with a valid string).
- `digest(data: bytes) -> bytes`: the 64-byte BLAKE2b of the package. A hasher uses it to reduce a
  big buffer (array values, the content of a `bytes`) to a fixed-width value **before** it enters the
  encoding, so the engine never copies a 50 MB preimage.

Rules of the contract (documented in `BaseHasher`, checked by one parametrised unit test over the
default registry, like `NATIVE_SAMPLES` in `equality`):

1. `encode` is **pure and cheap**: it does not call the registry, does not recurse, does not modify
   `data`, and does not read the clock, the environment or the filesystem (`ResolvedPathHasher`
   is the documented exception). It does not depend on `id()`, on `hash()` or on the iteration order
   of a set or of a mapping (an unordered `Node` says so instead).
2. It returns `bytes` (a leaf) or a `Node`. A leaf has no children: the engine digests it as is.
3. `Node.children` may contain any value, including a value built on the spot (a list of field
   names, a NumPy array, a `str` dtype): each child is encoded by **its own** hasher, found by the
   registry, so a user who replaces the hasher of `str` or of `ndarray` changes every container that
   holds one.
4. `ordered=True`: the position of a child matters. `ordered=False`: the children are cut into
   groups of `group` consecutive children (a mapping uses `group=2`: key, value), the digest of a
   group is the concatenation of the digests of its children, and the groups are sorted by digest.
   The result does not depend on the iteration order, and the keys need not be orderable.
5. The **tag** is a constant of the hasher, never derived from `type(data)`, except for the
   user-defined classes (dataclass, model, custom object, enum) whose qualified class name is the
   identity. A subclass therefore hashes like its base.
6. An exception propagates unchanged.

The shortest extension example has no registry and no `length` to remember:

```python
class PointHasher(BaseHasher[Point]):
    def encode(self, data):
        return Node("Point", [data.x, data.y])


register_hashers({Point: PointHasher()})
```

### The engine

```python
def digest_of(data, registry, ignore_unhashable):
    cache: dict[type, Hasher | None] = {}  # type -> hasher (None: placeholder), this call only
    memo: dict[int, tuple[object, bytes | None]] = {}  # id -> (object, digest); None = in progress

    def enter(obj):  # -> (frame, None) for a container, (None, digest) otherwise
        t = type(obj)
        if t not in cache:
            cache[t] = resolve(registry, t, ignore_unhashable)  # warns once per type
        hasher = cache[t]
        encoding = hasher.encode(obj) if hasher else tagged("unhashable", qualified_name(t))
        if type(encoding) is bytes:
            return None, blake2b(b"\x00" + encoding)
        seen = memo.get(id(obj))
        if seen is not None:
            if seen[1] is None:
                raise ValueError(f"Cannot hash a self-referencing {t!r}.")
            return None, seen[1]  # already hashed in this call (a shared container)
        memo[id(obj)] = (obj, None)  # keeps obj alive: its id cannot be reused in this call
        return Frame(obj, encoding, iter(encoding.children), []), None

    frame, result = enter(data)
    if frame is None:
        return result
    stack = [frame]
    while True:
        top = stack[-1]
        for child in top.children:  # resumes where it stopped
            child_frame, child_digest = enter(child)
            if child_frame is None:
                top.digests.append(child_digest)
            else:
                stack.append(child_frame)
                break
        else:  # children exhausted
            stack.pop()
            result = top.finish()  # groups, sort if unordered, BLAKE2b(0x01 + tag + 0 + digests)
            memo[id(top.obj)] = (top.obj, result)
            if not stack:
                return result
            stack[-1].digests.append(result)
```

Design points:

- **No `RecursionError`, no thread-local, no lock.** The state (stack, cache, memo) is local to the
  call, so threads and re-entrant calls (a hasher that calls `hash_object`) are independent.
  `HasherRegistry.hash` and `hash_object` delegate to this function; nothing else recurses.
- **Cycles and sharing in one dict.** `memo[id] = (obj, None)` while a container is in progress, its
  digest afterwards. Meeting an object in progress is a cycle; meeting a finished one is a shared
  container and costs a dictionary lookup. The object is kept in the entry so that the `id` of
  a temporary child (a dict of fields built by a hasher) cannot be reused by another object in the
  same call. Only containers are memoized, never leaves.
- **The cycle check is after `encode`**, so a leaf (a string, an array) never enters the memo, and a
  container is reported even if a custom hasher returns its own object as a child.
- **The per-call cache** removes the registry lookup per node. A hasher registered during a
  call is not seen by this call (the next call sees it).
- **`ignore_unhashable` lives here only.** The type is resolved once per call; a type with no hasher
  is replaced by `tagged("unhashable", qualified_name)` and a `UserWarning` is emitted **once per
  type per call**, with `stacklevel` pointing at the caller of `hash_object`. Without the
  option, the `TypeNotRegisteredError` carries the remedies (register a hasher, give the class a
  `__coola_hash__`, or pass the option).
- **Memory.** Besides the input, the engine holds the digests of the children of the containers
  being built (about 100 bytes per child of a node in progress) and the memo. A flat list of one
  million items needs about 100 MB for the digests. Streaming the digests of an ordered node into a
  running `blake2b` object would remove it; it is not done because the benchmark does not show a need.
- A hasher's exception reaches the caller with its own traceback and no frame of the package between,
  because the engine is a loop and not a recursion.

### Built-in hashers

| Hasher                                              | `encode(data)`                                                                                                                                                            | Default registration                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `ReprHasher`                                        | leaf; tag is the builtin type name (`bool`, `int`, `float`, `complex`), payload is the builtin `repr` (`-0.0` is `0.0`); a NumPy scalar is converted to the builtin first | `int`, `float`, `complex`, `bool`, `type(None)`, NumPy scalars        |
| `StringHasher`                                      | leaf `tagged("str", data)`; `UserString` is a leaf `tagged("UserString", str(data))`                                                                                      | `str`, `UserString`                                                   |
| `BytesHasher`                                       | leaf `tagged(tag, digest(raw))` with the tag `bytes`, `bytearray` or `memoryview`                                                                                         | `bytes`, `bytearray`, `memoryview`                                    |
| `EnumHasher`                                        | leaf `tagged("Enum", "module.Class.NAME")`; the member name, or the `repr` of the value for a combined flag                                                               | `Enum` (the enum-mixin rule is in the registry)                       |
| `DatetimeHasher`                                    | leaf; tag `date`, `datetime` or `time`; ISO form, aware values in UTC                                                                     | `date`, `datetime`, `time`                                            |
| `ReprHasher("timedelta")`, `StrHasher("UUID")`, ... | leaf with an **explicit** tag chosen at registration: `timedelta`, `timezone`, `UUID`, `ZoneInfo`                                                                         | `timedelta`, `timezone`, `UUID`, `ZoneInfo`                           |
| `DecimalHasher`, `FractionHasher`, `RangeHasher`    | leaf with the canonical form                                                                                                                                            | `Decimal`, `Fraction`, `range`                                        |
| `PathHasher`, `ResolvedPathHasher`                  | leaf `tagged("Path", posix form)`; the resolved variant is an opt-in                                                                                                      | `PurePath`                                                            |
| `SequenceHasher(tag)`                               | `Node(tag, data)`; a named tuple has the tag `tuple:Class(f1,f2)`                                                                                                         | `list`, `tuple`, `deque`, `Sequence`                                  |
| `MappingHasher(tag, ordered=False)`                 | `Node(tag, chain(k1, v1, k2, v2, ...), ordered, group=2)`                                                                                                                 | `dict`, `OrderedDict` (`ordered=True`), `Mapping`, `MappingProxyType` |
| `SetHasher(tag)`                                    | `Node(tag, data, ordered=False)`                                                                                                                                          | `set`, `frozenset`                                                    |
| `DataclassHasher`                                   | `Node("dataclass:module.Class", chain(name, value, ...), ordered=False, group=2)` over the fields with `compare=True`                                                     | fallback: `dataclasses.is_dataclass`                                  |
| `ContentHasher`                                     | `Node("custom:module.Class", [data.__coola_hash__()])`                                                                                                                    | fallback: the type has a `__coola_hash__` method                      |
| NumPy, torch, JAX, pandas, polars, pyarrow, xarray  | a leaf `tagged(tag, header + digest(raw values))` for numeric buffers, otherwise a `Node` whose children are plain values (names, dtype strings, arrays, lists)           | lazy groups                                                |
| `PydanticModelHasher`, `SecretHasher(on_secret)`    | `Node("pydantic:module.Class", chain(field, value, ...), ordered=False, group=2)`; a secret is a leaf or an error                                                         | lazy group                                                 |

There are no public leaf functions such as `hash_string` or `hash_bytes`: `hash_object("x")` is the
supported way to hash a value, and `tagged` and `digest` are the building blocks of a hasher. The
registry API is `HasherRegistry` with `register`, `register_many`, `register_fallback`,
`register_lazy`, `find_hasher`, `has_hasher`, `copy`, `get_default_registry` (and its alias
`get_default_hasher_registry`), `get_stable_registry` and `register_hashers`. `Hasher` and
`BaseHasher` follow the family convention (`Transformer`/`BaseTransformer`,
`Summarizer`/`BaseSummarizer`).

### The encoding

One rule for every hash, written once in `engine.py` and `base.py`:

```
digest(x)  = BLAKE2b-512(preimage, person = b"coola-hash-1")           # 64 bytes, always
leaf       : preimage = 0x00 + tag + 0x00 + payload                     # what `encode` returned
node       : preimage = 0x01 + tag + 0x00 + e1 + e2 + ... + en          # one entry per child
entry      : 0xFF + digest (64 bytes)  |  size (1 byte) + leaf           # a leaf of at most 64 bytes is inlined
unordered  : the groups of `group` children are concatenated, then sorted, then joined
hash_object(x, length=n) = digest(x).hex()[:n]                         # n characters
```

- **Self-delimiting entries** make the concatenation unambiguous without delimiters: an entry is
  either a marker and a fixed-width digest, or a size byte and the leaf. The leading `0x00`/`0x01` byte
  makes a leaf and a node with the same tag different by construction. Inlining the short leaves
  makes a flat container one BLAKE2b call instead of one per item.
- **`length` is chosen by the caller and cuts the end.** The caller asks for 2 to 128 characters.
  Every digest is a full 64-byte BLAKE2b, so the hash of any length is the start of the
  longest one, a hasher never sees `length`, and `length` is validated once, before the traversal.
  The default `length=64` is the first half of the digest.
- **The format version** is the `person` parameter of BLAKE2b (at most 16 bytes, free): the digests
  of two formats can never be mistaken for one another, and a change of the encoding is one
  constant (for example `coola-hash-2`), plus the regenerated golden vectors.
- **Tag rule.** A scalar and a stdlib container have a **constant** tag (`int`, `str`, `list`,
  `OrderedDict`, `date`, `UUID`, `Decimal`, `Path`, `numpy.ndarray`, ...), so a subclass hashes like
  its base and a library moving a class to another module does not change a hash. A user-defined
  type (dataclass, model, enum, custom object) uses `module.qualname`, because the class **is** its
  identity. A tag is never derived from `type(data)` anywhere else. A named tuple adds its class
  name and field names to the tag of `tuple` (so it differs from a plain tuple and from another
  named tuple).
- **Big payloads** are reduced with `digest()` by the hasher (arrays, `bytes`), so the preimage of a
  leaf is a few hundred bytes at most.

### Registry and dispatch

`HasherRegistry(TypeRegistry[Hasher])` keeps its role: map types to hashers and find the most
specific one along the MRO, then the fallbacks (registration order), then `object` if one is
registered (the default registry has none: an unsupported type fails loudly). It keeps
`find_hasher` and `has_hasher` and gets **one** more rule, in `find_hasher`:

> A subclass of `Enum` resolves to the hasher of `Enum` unless a hasher is registered for the enum
> class itself (or for a parent that is not a builtin base of the mixin).

`StringHasher` and `ReprHasher` have no `isinstance(data, Enum)` branch, and `HasherRegistry.hash` is
`digest_of(...).hex()[:length]` with the validation of `length`. The registry has no knowledge of
cycles, placeholders or the recursion limit.

`get_default_registry()` is the singleton with the built-in hashers, the entries of the plugin group
`coola.hashing.hashers` and the lazy groups; `get_stable_registry()` is the same without the plugins
and is the registry of `generate_stable_content_id` and `generate_stable_uuid5`.
`register_hashers(mapping, *, exist_ok=False)` is keyword-only for `exist_ok`.

### Library hashers

The hashers of the optional libraries have no registry plumbing. They return a `Node` whose children
are plain values, or a leaf; the engine hashes each child with the hasher of its type:

| Library | `encode(data)`                                                                                                                                                                                                                            |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NumPy   | numeric dtype: leaf `tagged("numpy.ndarray", f"{dtype}\0{shape}\0" + digest(canonical bytes))` (masked arrays add the digest of the mask); `object` dtype: `Node("numpy.ndarray", [dtype, shape, items])`                                 |
| torch   | leaf as NumPy with the tag `torch.Tensor`, the dense CPU values, no device and no `requires_grad`                                                                                                                                         |
| JAX     | leaf as NumPy with the tag `jax.Array`                                                                                                                                                                                                    |
| pandas  | `Node("pandas.DataFrame", [columns key, index key, dtype and values of each column])`; a key is `(names, dtype, values)`; a categorical dtype adds its categories and `ordered`; a missing value is `None`; a `NaN` name is `None` |
| polars  | `Node("polars.DataFrame", [name, str(dtype), values, ...])` per column; numeric columns without nulls pass the NumPy array                                                                                                                |
| pyarrow | the fixed-width types as NumPy leaves with the digest of the validity bitmap; the other types as a `Node` over `to_pylist()`                                                                                                              |
| xarray  | `Node("xarray.DataArray", [name, dims, values, coords, attrs])`                                                                                                                                          |

The semantics of every library hasher follow the rules of the guide: dtype, shape and values count;
the memory layout does not; `-0.0` and NaN are canonical. The helpers that three modules share
(`canonicalize_floats`, `hash_array_content`) live in `coola.hashing.numpy`.

### Custom objects: `__coola_hash__`

```python
class SupportsCoolaHash(Protocol):
    def __coola_hash__(self) -> object: ...
```

A class that cannot be a dataclass says **what its content is** and the engine hashes it (with the
tag `custom:module.Class`, so two classes with the same content differ):

```python
class Point:
    def __coola_hash__(self):
        return (self.x, self.y)
```

The object does not recurse, so it cannot bypass the cycle check, nobody forwards `length`, and the
fallback predicate is `callable(getattr(t, "__coola_hash__", None))` with no signature inspection. The dunder
name cannot collide with a method of a library (`polars.Series.hash`). A dataclass keeps the
dataclass hasher even if it defines `__coola_hash__` (type > fallback order, then registration
order of the fallbacks).

### Secrets and pydantic

Secrets become **types with a hasher**, so the policy is the one of the registry and applies
wherever a secret is, not only in the fields of the model:

```python
class SecretHasher(BaseHasher[SecretStr | SecretBytes]):
    def __init__(self, on_secret: Literal["error", "reveal", "exclude"] = "error"): ...

    # "error":   ValueError, the message never contains the secret
    # "reveal":  leaf tagged("pydantic.SecretStr", value)
    # "exclude": leaf tagged("pydantic.SecretStr")   (the value is not part of the hash)
```

`PydanticModelHasher.encode` is `Node("pydantic:module.Class", fields and extras)`, nothing else:
there is no second traversal to unwrap secrets. The default registry registers
`SecretHasher("error")` for `SecretStr` and `SecretBytes` in the pydantic lazy group. To reveal or
exclude, a caller builds a registry:

```python
registry = get_default_registry().copy()
registry.register_many(get_secret_hashers("reveal"), exist_ok=True)  # SecretStr and SecretBytes
hash_object(model, registry=registry)
```

A secret in a dict or in a dataclass is governed by the same policy as in a model. `"exclude"` does
not remove the key or the item: it replaces the value by a constant (the position of a secret is part
of the schema of the class, so it is not information; the value is never hashed).

### Errors and options

| Situation                                   | Behavior                                                                                                         |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `length` is not an even `int` from 2 to 128 | `ValueError` before any traversal; a `bool` or a `float` is rejected with the same message                       |
| No hasher for a type                        | `TypeNotRegisteredError` (a `KeyError`) with the remedies, or a lossy placeholder and one `UserWarning` per type |
| A container that contains itself            | `ValueError("Cannot hash a self-referencing <type>.")`                                                           |
| Nesting of any depth                        | works; the only limit is memory (no `RecursionError`, no `ValueError`)                                           |
| A secret with `on_secret="error"`           | `ValueError`, without leaking the value                                                                          |
| An exception raised by a hasher             | propagates unchanged                                                                                             |

## Public API

```python
from coola.hashing import (
    hash_object,  # (data, *, registry=None, length=64, ignore_unhashable=False)
    HasherRegistry,
    get_default_registry,
    get_default_hasher_registry,  # alias
    get_stable_registry,
    register_hashers,  # (mapping, *, exist_ok=False)
    Hasher,  # protocol: encode() only
    BaseHasher,
    Node,
    tagged,
    digest,
    SupportsCoolaHash,
    ContentHasher,
    DataclassHasher,
    SequenceHasher,
    MappingHasher,
    SetHasher,
    StringHasher,
    BytesHasher,
    EnumHasher,
    ReprHasher,
    StrHasher,
    DatetimeHasher,
    PathHasher,
    ResolvedPathHasher,
    # lazy
    NumpyArrayHasher,
    TorchTensorHasher,
    JaxArrayHasher,
    PandasSeriesHasher,
    PandasDataFrameHasher,
    PolarsSeriesHasher,
    PolarsDataFrameHasher,
    PyarrowArrayHasher,
    PyarrowTableHasher,
    XarrayHasher,
    PydanticModelHasher,
    SecretHasher,
    get_secret_hashers,
)
```

`hash_object(data, *, registry=None, length=64, ignore_unhashable=False)` has no positional option.
`HasherRegistry.hash(data, *, length=64, ignore_unhashable=False)` is the registry-level entry point
(it is what the guide and the doctests use) and calls the same engine. The decimal, fraction and range
hashers live in `coola.hashing.canonical` (public but not re-exported).

## Dependents

| Package                                 | Use                                                                                                                                                  |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `identifier.generate_stable_content_id` | `hash_object(data, registry=get_stable_registry(), length=...)`. Every identifier changes with the format version.                                      |
| `identifier.generate_stable_uuid5`      | Hashes at `length=128`, the whole digest.                                                                                                            |
| `equality`                              | The property "equal implies same hash" is tested from `tests/properties`; the known gaps are listed in the test matrix.                                |
| User code with a custom hasher          | Implements `encode`. A class that cannot be a dataclass defines `__coola_hash__`.                                                                     |
| Plugins of `coola.hashing.hashers`      | Their hashers implement `encode`.                                                                                                                    |

## Design choices and trade-offs

**One contract: `encode` returns a description, the engine hashes.** It is the only way to make
cycles, depth, sharing, `length` and the placeholder properties of the engine instead of the
discipline of every hasher, and it is what `equality`, `summary` and `recursive` do. The alternative,
a context object passed to a `hash(data, ctx)` method, removes the repeated parameters but a hasher
still recurses, so the cycle protection stays per hasher and the engine cannot be an explicit loop.

**The caller chooses the length of the hash.** `hash_object(x, length=16)` returns 16 characters (an
even `int` from 2 to 128). Internally one 64-byte digest is computed and the first `length` characters
are kept: no `size` argument in the engine, no special case for the root. A side effect is that the
hash of a shorter length is the start of the longer one, so a short id can be matched against a stored
full hash with `startswith`. Computing the root with `digest_size = length // 2` gives independent
values at each length, at the cost of one more argument in the engine.

**Shared containers are memoized.** Unlike `recursive_apply`, a hash is pure, so the memo is
unobservable, and it turns a DAG from exponential to linear. The cost is one dictionary entry per
container and the objects kept alive until the call ends, which also makes the `id` reuse of temporary
children impossible. Without the memo, a plain set of ancestors detects cycles with three lines less,
but 2^20 paths take 4 s.

**A cycle is an error.** `objects_are_equal` accepts a cyclic structure and says "equal", so strictly
"equal implies the same hash" is violated by a cycle. A deterministic back-reference (the distance to
the ancestor) would make a cycle hashable, but the digest of a container in a cycle would then depend
on where it is reached from, which breaks the memo and the "hash of a child is reusable" property. A
cycle in a hashed configuration is almost always a bug, so the loud error is the better default.

**A custom object says what its content is.** `__coola_hash__(self) -> object` returns the content;
the object does not recurse, there is no signature inspection, and the name cannot collide with a
library method such as `polars.Series.hash`. The alternative, `hash(registry, length,
ignore_unhashable)`, forces the engine to support a re-entrant, parameterised call.

**Secrets are types with a hasher, and `exclude` replaces instead of removing.** There is no second
traversal to unwrap secrets, and the policy is the one of the registry, so a caller that wants
`reveal` builds a registry once. Keeping an `on_secret` argument on the model hasher would treat a
secret in a dict differently from a secret in a model.

**Short leaves are inlined in the preimage of their parent.** A flat container of short items makes one
BLAKE2b call, which is about 4 times faster than one call per item. The price is two forms of child
entry (a size byte and the leaf, or a marker and a digest). Streaming the digests of an ordered node
into a running `blake2b` object (memory) is the optimisation that is held back until a benchmark or a
user shows that hashing a million-element list is on a hot path.

**No public leaf functions.** A function such as `hash_string` hashes one value under a tag with a
`length`, which is `hash_object` for a scalar, and it keeps `length` in the leaf code. A caller that
wants a plain BLAKE2b of a string uses `hashlib`. `tagged` and `digest` are public because a custom
hasher needs them.

**A format version in the digest.** `person=b"coola-hash-1"` costs nothing, makes two formats
impossible to confuse (a stored hash of another format never equals a new one by accident), and turns
a deliberate format change into a one-line constant plus the golden vectors. Without it, only the
golden vectors would signal a format change.

**The stable registry and the plugin group.** The plugin group is how a library adds its types, and
`get_stable_registry()` keeps the identifiers independent of the plugins.

**Only `compare=True` fields of a dataclass.** It is the rule of `objects_are_equal`, so equal
dataclasses have the same hash, and a cache or a lazily computed field cannot make two equal objects
different. A user who wants every field registers a hasher of their own.

## Not covered

- **Hashes of types that equality supports and hashing does not.** The next candidates are `Index`, `MultiIndex` and `Categorical` as values, `np.datetime64`,
  `np.timedelta64`, `array.array`, `dict_keys`, `dict_values`, `dict_items`, `slice`, `Ellipsis`,
  `re.Pattern`, `torch.dtype` and `torch.device`. Each is one hasher and one registration.
- **Functions, classes and other callables** are not hashed (a class in a configuration needs its
  own hasher). A `type` hasher by qualified name is a one-line registration if a user asks.
- **Hashing the content of a file** (`ResolvedPathHasher` hashes the path, not the file).
- **A cyclic structure** and **a structure that changes during the call** (no lock).
- **Streaming and parallel hashing**: the engine is a single loop; a huge array is already one
  call of BLAKE2b on a buffer.
- **Meta and float8 tensors.**

# `coola.hashing`: target state and the changes to reach it

This document describes the design that `coola.hashing` should have, then the changes to make to the
current implementation (`main`, `d2e60e6fc`) to get there. The design follows the rules of the user
guide (`docs/docs/uguide/hashing.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: registry dispatch, MRO lookup, lazy imports, strict contracts,
"equality drives hashing"), and it stays aligned with the sibling packages `equality`, `summary`,
`recursive` and `iterator` (`review/equality.md`, `review/summary.md`, `review/recursive.md`). Its
aim is a small package that is easy to maintain: **one engine, one extension point, one encoding
rule**. Breaking changes are accepted, and the future state is more important than the past one.

Evidence comes from probes and a throw-away prototype run on `main` (Python 3.14, scripts kept
outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from the
code. Nothing in `src`, `tests` or `docs` was changed.

## Constraints

1. **Simple and easy to maintain.** The package is 3,686 lines, 25 modules and 31 `hash` methods
   that all repeat the same four parameters (and 30 `# noqa: ARG002` to silence the ones they do
   not use). The target removes the code that exists only to work around the design (the
   thread-local cycle state, two copies of it, the `RecursionError` translation, the signature
   inspection of `_has_hash_method`, the secret traversal of `unwrap_secrets`, the positional-argument
   shim) instead of adding code next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. **Every hash value changes** (the encoding changes, section
   3.4), so every stable identifier changes too. The release notes say so; the golden vectors are
   regenerated once, deliberately.
3. **Consistent with equality.** "Equality drives hashing, not the reverse": equal objects must have
   the same hash, and where a hasher disagrees with `objects_are_equal` the hasher is changed
   (F7, F8).
4. **No slower than today, and the gate exists.** `tests/benchmarks/test_hashing_benchmark.py` was
   added with this document, and the baseline of `main` is saved (appendix). A design that is
   simpler but slower is rejected (section 8).
5. **Consistent with the siblings.** The same input is handled the same way by `equality`,
   `summary`, `recursive`, `iterator` and `hashing`: dataclasses are entered, cycles are detected,
   there is no `RecursionError`, options are keyword-only, the engine owns the invariants and the
   type-specific class only describes its type.
6. **Optional libraries stay lazy.** The package imports no optional library; the hashers of
   NumPy, torch, JAX, pandas, polars, pyarrow, xarray and pydantic are loaded the first time the
   registry looks for one of their types (`register_lazy`, unchanged).

## 1. Properties of the sub-package

A hash is a short text that identifies the **content** of a (possibly nested) object, to detect a
duplicate, to key a cache or to build a stable identifier. It is a fingerprint, **not** a
cryptographic guarantee (no passwords, signatures or integrity against an attacker). These are the
rules of the user guide. Every design choice below exists to make one of them true **by
construction** (in the engine, in the encoding) instead of by the discipline of each hasher.

| Property                 | Statement                                                                                                                                                                                                                    | Enforced by                               |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Equality-consistent      | Objects that are equal for `objects_are_equal` have the same hash. The reverse can fail for the types that share a hash on purpose (a `list` subclass and a `list`, a NumPy `float64` and a `float`).                        | each hasher; the property tests           |
| Deterministic            | The same value gives the same hash across calls, processes, machines and operating systems. No dependence on `PYTHONHASHSEED`, `id()`, the iteration order of a set, the locale, the clock or the time zone.                 | the encoding; golden vectors              |
| Type-sensitive           | Values of different types hash differently, even if their text is the same: `1`, `1.0`, `True`, `"1"`, `b"1"`, `memoryview(b"1")` and `[1]` are all distinct. Subclasses hash like their base (documented).                  | a constant tag per hasher                 |
| Structure-sensitive      | Nesting and item boundaries matter: `[[1], 2]`, `[1, [2]]` and `[1, 2]` differ, and so do `["ab", "c"]` and `["a", "bc"]`.                                                                                                   | the node encoding (fixed-width digests)   |
| Order rules              | Ordered containers are order-sensitive; unordered ones (`dict`, `set`, `frozenset`) are not. `OrderedDict` is ordered, like in `objects_are_equal`.                                                                          | `Node(ordered=...)`                       |
| Composable               | The hash of a container is computed only from the digests of its children, resolved through the registry, so each type is extended on its own and `hash_object` of a child is reusable.                                      | the engine                                |
| Fixed width              | The output is a lowercase hex string of `length` characters (an even `int` from 2 to 128, default 64), and `hash_object(x, length=n)` is the first `n` characters of `hash_object(x, length=128)`.                           | the engine (one digest, truncated at end) |
| No silent false equality | An unsupported type raises `TypeNotRegisteredError` (a `KeyError`). `ignore_unhashable=True` is the explicit, lossy opt-out and warns once per type.                                                                         | the engine                                |
| Cycle-safe               | A container that contains itself raises `ValueError`, whichever hasher is registered for it.                                                                                                                                 | the engine                                |
| Any depth                | No `RecursionError`, whatever the nesting or the caller's stack: the engine uses an explicit stack. The depth is bounded by memory only.                                                                                     | the engine                                |
| Linear cost              | The cost is linear in the number of **objects**, not of paths: a container reachable by several paths (a DAG) is hashed once per call. A hash is pure, so the memoization is invisible.                                      | the engine (per-call memo)                |
| Pure, thread-safe        | Hashing has no side effect and keeps no global state (no thread-local, no lock): a registry and a hasher can be shared by threads and re-entrant calls. Reading the filesystem is an explicit opt-in (`ResolvedPathHasher`). | the engine, stateless hashers             |
| Errors propagate         | An exception of a hasher propagates unchanged. Invalid `length` raises `ValueError` before any traversal. A secret is never hashed unless asked for, wherever it is (section 3.8).                                           | the engine, `SecretHasher`                |
| Consistent with siblings | Dataclasses and pydantic models are entered, only their `compare=True` fields count (like `objects_are_equal`); options are keyword-only; one extension point.                                                               | the default registry                      |

Not guaranteed, and documented as such:

- **Stability across versions of `coola`.** The format can change in a major release. The golden
  vectors make a change deliberate, the format version (section 3.4) makes it impossible to confuse
  two formats, and `get_stable_registry()` makes the stable identifiers independent of the installed
  plugins.
- **Stability across versions of a library** for the types of that library when the library changes
  what an object means (the dtype names of pandas, for example).
- **Hashes of classes that are renamed or moved.** A dataclass, a model, an enum and a custom object
  hash with their qualified class name.
- **Cycles.** A cyclic structure raises `ValueError`, although `objects_are_equal` accepts it
  (decision D4).
- **Resistance to an adversary.** Do not use a hash as a security boundary.

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Status         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The protections live in the registry, the recursion lives in each hasher.** Every hasher calls `registry.hash` for its children. The cycle check (a thread-local `active` set) and the `RecursionError` translation are in `HasherRegistry.hash`; `PydanticModelHasher` needs a second thread-local set of its own because its recursion goes through `unwrap_secrets` callbacks. A custom hasher that does not forward `registry` bypasses all of it. The class that dispatches by type also owns the cycle guard, the placeholder and the enum rule (90 lines). | **[reading]**  |
| F2  | **Depth.** The recursion costs two frames per level: a list nested **497** levels deep raises `ValueError: Cannot hash data that is nested too deeply` (a dict: also 497; the interpreter limit is 1,000). A `ValueError` is also the type of the self-reference error, so the caller cannot tell the two apart. The explicit stack of the prototype hashes 100,000 levels in 0.18 s.                                                                                                                                                                               | **[verified]** |
| F3  | **A DAG costs one visit per path.** `x = [x, x]` repeated 20 times has 20 objects and 2^20 paths: 4.2 s (0.26 s at 16 levels). Hashing is pure, so a container that was already hashed in the call can be reused (the prototype does 60 levels in 0.1 ms).                                                                                                                                                                                                                                                                                                          | **[verified]** |
| F4  | **`UserString` cannot be hashed.** It is a `Sequence` whose items are `UserString` again, so `SequenceHasher` recurses until the limit and the user gets "nested too deeply". The same trap exists for any `Sequence` of itself.                                                                                                                                                                                                                                                                                                                                    | **[verified]** |
| F5  | **The hasher contract is heavy.** `hash(data, registry, *, length, ignore_unhashable) -> str`: the 31 implementations repeat four parameters, 30 `# noqa: ARG002` silence the ones they do not use, and a hasher that recurses must remember to forward two of them. `length` is applied to **every node**, so `hash_object(x, length=16)` is not a prefix of `hash_object(x)` and a leaf function (`hash_string`, `hash_tagged`, `hash_bytes`, `hash_enum`, `hash_path`) each validates it again.                                                                  | **[reading]**  |
| F6  | **Speed.** 1.4 µs per item of a list and 3.1 µs per item of a dict (100,000 items: 141 ms and 312 ms; a set: 164 ms). A fresh `blake2b` per leaf and per container, a hex round trip for each digest, a registry lookup per node (no per-call cache). There was no hashing benchmark before this document.                                                                                                                                                                                                                                                          | **[verified]** |
| F7  | **Equal objects with different hashes.** A dataclass with a `compare=False` field (equal for `objects_are_equal`, hash differs); a pandas `Series` named `None` and one named `NaN` (equal, hash differs). This is the open item 14.7 of `suggestions.md` (22.9); no test covers it today. `MappingProxyType` is equal to an equal proxy but cannot be hashed (14.4).                                                                                                                                                                                               | **[verified]** |
| F8  | **Type-sensitivity gaps.** A `memoryview` hashes like a `bytes` with the same content (same tag, and `objects_are_equal` says they differ). The tags have no rule: `ReprHasher` uses the module-qualified class name, `StrHasher` the bare one (`str:UUID`, so a user class named `UUID` registered with `StrHasher` collides with `uuid.UUID`), `DatetimeHasher` the bare `__qualname__`, `SequenceHasher` a free string.                                                                                                                                          | **[verified]** |
| F9  | **The enum-mixin rule is written three times.** `(str, Enum)`, `(int, Enum)` and `(bytes, Enum)` members would resolve to the hasher of the base type: `HasherRegistry._find_enum_hasher`, `StringHasher.hash` and `ReprHasher.hash` each handle it.                                                                                                                                                                                                                                                                                                                | **[reading]**  |
| F10 | **`SupportsHash` re-enters the registry and needs a hack to be found.** An object implements `hash(registry, length, ignore_unhashable)` and calls `registry.hash` itself. The fallback predicate `_has_hash_method` binds the signature to tell it from `polars.Series.hash(seed=...)` and every other `hash` method of the ecosystem.                                                                                                                                                                                                                             | **[reading]**  |
| F11 | **Secrets: a second traversal.** `unwrap_secrets` walks `dict`, `list`, `tuple`, `set` and `frozenset` (exact types only) to drop or unwrap secrets, with an `on_model` callback and its own thread-local cycle set. The policy is an argument of the **model** hasher, so a model nested in a dataclass uses the default (`hash_object(H(M(s="x")))` raises `ValueError`), and a bare `SecretStr` in a dict raises `TypeNotRegisteredError`.                                                                                                                       | **[verified]** |
| F12 | **The surface is large.** 39 public names, 25 modules, and five public leaf functions (`hash_string`, `hash_bytes`, `hash_tagged`, `hash_enum`, `hash_path`) that duplicate what `hash_object` does with a tag. `hash_object` keeps a deprecated positional form (two overloads and `_resolve_legacy_arguments`, 50 lines), `register_hashers(mapping, exist_ok)` takes `exist_ok` by position.                                                                                                                                                                     | **[reading]**  |
| F13 | **The documentation is stale or wrong in places.** The module docstring of `interface.py` says "recursively apply a function to all items in nested data"; the comment above `dict` in `_register_default_hashers` says "recursive transformation of keys and values"; `HasherRegistry.hash` documents `KeyError` (it raises `TypeNotRegisteredError`, a subclass) and "nested too deeply for the recursion limit" as a `ValueError`; the guide teaches the deprecated positional form.                                                                             | **[reading]**  |
| F14 | **`unhashable` placeholders warn once per object.** `hash_object([A() for _ in range(5)], ignore_unhashable=True)` emits five warnings (the default filter shows one per call site, but a `-W always` log is flooded). The warning points at coola's frame, not the caller's.                                                                                                                                                                                                                                                                                       | **[verified]** |
| F15 | **`length` is not validated as an `int`.** `length=2.0` raises a raw `TypeError` from the integer division, and `length=True` is rejected with the message "got True".                                                                                                                                                                                                                                                                                                                                                                                              | **[verified]** |

What is already right and must not regress: type dispatch with MRO, the fallbacks of
`TypeRegistry` and the lazy groups; the tagged, fixed-width Merkle composition (a container hash
depends only on the digests of the children); the canonical scalars (`-0.0` is `0.0`, one NaN, aware
datetimes and times in UTC, `Decimal` by `as_tuple`, `Fraction` in lowest terms, `range` in constant
time, NumPy scalars like the builtin, huge integers independent of `sys.set_int_max_str_digits`, lone
surrogates); the array rules (dtype, shape and values, not the layout, the device or `requires_grad`);
the stable registry; independence of `PYTHONHASHSEED` **[verified]**; thread safety of concurrent
calls **[verified]**; a hashing call of 50 MB of bytes in 0.09 s **[verified]**.

## 3. Architecture

```
hash_object(data, *, registry=None, length=64, ignore_unhashable=False)
        |   validates length; one registry (default: the singleton); one call = one engine run
        v
digest_of(data, registry, ignore_unhashable)           coola/hashing/engine.py   (about 60 lines)
        |   explicit stack of frames; per-call type -> hasher cache; per-call memo / ancestors
        |
        |-- hasher   = registry.find_hasher(type(data))        MRO, fallbacks, Enum rule
        |-- encoding = hasher.encode(data)                     pure: no registry, no recursion
        |        bytes  -> leaf: digest = BLAKE2b(0x00 + encoding)
        |        Node   -> push the children; when they are done:
        |                  digest = BLAKE2b(0x01 + tag + 0x00 + digests of the children)
        v
the 64-byte digest of the root  ->  hex()[:length]
```

The engine owns the invariants (dispatch cache, composition, order rule, cycles, depth, sharing,
placeholder). A hasher **only describes the content of a type**: a tagged byte string for a leaf, or
a tag and the children for a container. That is the same split as `equality` (the engine owns
identity, type, depth and cycles; a tester returns the child pairs) and `recursive` (a transformer
returns the children and how to rebuild). `registry`, `length` and `ignore_unhashable` are no longer
arguments of a hasher, so a hasher cannot forget a protection, cannot recurse and cannot leak a
parameter.

### 3.1 The hasher contract

```python
@dataclass(slots=True)
class Node:
    tag: str  # constant of the hasher (section 3.4)
    children: Iterable[Any]  # the values to hash, consumed once, lazily
    ordered: bool = True  # False: the groups of children are sorted by digest
    group: int = 1  # number of consecutive children that form one unordered entry


Encoding = bytes | Node


class Hasher(Protocol[T_contra]):
    def encode(self, data: T_contra) -> Encoding:
        """A leaf (``tagged(tag, payload)``) or a ``Node`` that describes the content of ``data``."""


class BaseHasher(ABC, Generic[T]):  # convenience base class, as today
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
   identity (section 3.4). A subclass therefore hashes like its base.
6. An exception propagates unchanged.

The extension example of the guide becomes the shortest possible one, with no registry and no
`length` to remember:

```python
class PointHasher(BaseHasher[Point]):
    def encode(self, data):
        return Node("Point", [data.x, data.y])


register_hashers({Point: PointHasher()})
```

### 3.2 The engine

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
  container and costs a dictionary lookup (F3). The object is kept in the entry so that the `id` of
  a temporary child (a dict of fields built by a hasher) cannot be reused by another object in the
  same call. Only containers are memoized, never leaves.
- **The cycle check is after `encode`**, so a leaf (a string, an array) never enters the memo, and a
  container is reported even if a custom hasher returns its own object as a child.
- **The per-call cache** removes the registry lookup per node (F6). A hasher registered during a
  call is not seen by this call (the next call sees it).
- **`ignore_unhashable` lives here only.** The type is resolved once per call; a type with no hasher
  is replaced by `tagged("unhashable", qualified_name)` and a `UserWarning` is emitted **once per
  type per call**, with `stacklevel` pointing at the caller of `hash_object` (F14). Without the
  option, the `TypeNotRegisteredError` carries the remedies (register a hasher, give the class a
  `__hash_content__`, or pass the option).
- **Memory.** Besides the input, the engine holds the digests of the children of the containers
  being built (about 100 bytes per child of a node in progress) and the memo. A flat list of one
  million items needs about 100 MB for the digests, like today's hex strings. Streaming the digests
  of an ordered node into a running `blake2b` object would remove it; it is a three-line change if
  the benchmark shows a need.
- `func` does not exist: a hasher's exception reaches the caller with its own traceback and no frame
  of the package between (the recursion of today adds two per level).

### 3.3 Built-in hashers

| Hasher                                              | `encode(data)`                                                                                                                                                                | Default registration                                                       |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `ReprHasher`                                        | leaf; tag is the builtin type name (`bool`, `int`, `float`, `complex`), payload is the builtin `repr` (`-0.0` is `0.0`); a NumPy scalar is converted to the builtin first     | `int`, `float`, `complex`, `bool`, `type(None)`, NumPy scalars             |
| `StringHasher`                                      | leaf `tagged("str", data)`; `UserString` is a leaf `tagged("UserString", str(data))` (F4)                                                                                     | `str`, `UserString`                                                        |
| `BytesHasher`                                       | leaf `tagged(tag, digest(raw))` with the tag `bytes`, `bytearray` or `memoryview` (F8)                                                                                        | `bytes`, `bytearray`, `memoryview`                                         |
| `EnumHasher`                                        | leaf `tagged("Enum", "module.Class.NAME")`; the member name, or the `repr` of the value for a combined flag                                                                   | `Enum` (the enum-mixin rule is in the registry, section 3.5)               |
| `DatetimeHasher`                                    | leaf; tag `date`, `datetime` or `time`; ISO form, aware values in UTC (unchanged rules, 22.7 is fixed)                                                                        | `date`, `datetime`, `time`                                                 |
| `ReprHasher("timedelta")`, `StrHasher("UUID")`, ... | leaf with an **explicit** tag chosen at registration (F8): `timedelta`, `timezone`, `UUID`, `ZoneInfo`                                                                        | `timedelta`, `timezone`, `UUID`, `ZoneInfo`                                |
| `DecimalHasher`, `FractionHasher`, `RangeHasher`    | leaf with the canonical form (unchanged)                                                                                                                                      | `Decimal`, `Fraction`, `range`                                             |
| `PathHasher`, `ResolvedPathHasher`                  | leaf `tagged("Path", posix form)`; the resolved variant is an opt-in                                                                                                          | `PurePath`                                                                 |
| `SequenceHasher(tag)`                               | `Node(tag, data)`; a named tuple has the tag `tuple:Class(f1,f2)`                                                                                                             | `list`, `tuple`, `deque`, `Sequence`                                       |
| `MappingHasher(tag, ordered=False)`                 | `Node(tag, chain(k1, v1, k2, v2, ...), ordered, group=2)`                                                                                                                     | `dict`, `OrderedDict` (`ordered=True`), `Mapping`, `MappingProxyType` (F7) |
| `SetHasher(tag)`                                    | `Node(tag, data, ordered=False)`                                                                                                                                              | `set`, `frozenset`                                                         |
| `DataclassHasher`                                   | `Node("dataclass:module.Class", chain(name, value, ...), ordered=False, group=2)` over the fields with `compare=True` (F7)                                                    | fallback: `dataclasses.is_dataclass`                                       |
| `ContentHasher`                                     | `Node("custom:module.Class", [data.__hash_content__()])` (section 3.7)                                                                                                        | fallback: the type has a `__hash_content__` method                         |
| NumPy, torch, JAX, pandas, polars, pyarrow, xarray  | a leaf `tagged(tag, header + digest(raw values))` for numeric buffers, otherwise a `Node` whose children are plain values (names, dtype strings, arrays, lists) (section 3.6) | lazy groups, unchanged                                                     |
| `PydanticModelHasher`, `SecretHasher(on_secret)`    | `Node("pydantic:module.Class", chain(field, value, ...), ordered=False, group=2)`; a secret is a leaf or an error (section 3.8)                                               | lazy group, unchanged                                                      |

Removed: the five public leaf functions of F12 (`hash_string`, `hash_bytes`, `hash_tagged`,
`hash_enum`, `hash_path`: `hash_object("x")` is the supported way; `tagged` and `digest` are the
building blocks of a hasher), `hash_pydantic_model` and `unwrap_secrets` (section 3.8),
`SupportsHash` and `HashableHasher` (replaced by `__hash_content__` and `ContentHasher`), the
`registry`, `length` and `ignore_unhashable` parameters of every hasher, the positional form of
`hash_object` and of `register_hashers`. Kept as they are: `Hasher` and `BaseHasher` (the family
convention: `Transformer`/`BaseTransformer`, `Summarizer`/`BaseSummarizer`), `HasherRegistry` with
`register`, `register_many`, `register_fallback`, `register_lazy`, `find_hasher`, `has_hasher`,
`copy`, `get_default_registry` and its alias `get_default_hasher_registry`, `get_stable_registry`,
`register_hashers`.

### 3.4 The encoding

One rule for every hash, written once in `engine.py` and `base.py`:

```
digest(x)  = BLAKE2b-512(preimage, person = b"coola-hash-1")           # 64 bytes, always
leaf       : preimage = 0x00 + tag + 0x00 + payload                     # what `encode` returned
node       : preimage = 0x01 + tag + 0x00 + d1 + d2 + ... + dn          # fixed-width children
unordered  : the groups of `group` children are concatenated, then sorted, then joined
hash_object(x, length=n) = digest(x).hex()[:n]
```

- **Fixed-width children** make the concatenation unambiguous without delimiters. The leading
  `0x00`/`0x01` byte makes a leaf and a node with the same tag different by construction.
- **One digest size.** Every node is a full 64-byte digest and `length` only truncates the root: the
  hashes of different `length` are prefixes of one another, a hasher never sees `length`, and
  `length` is validated once (decision D2). The default `length=64` is the first half of the digest.
- **The format version** is the `person` parameter of BLAKE2b (at most 16 bytes, free): the digests
  of two formats can never be mistaken for one another, and a change of the encoding is one
  constant, `coola-hash-2`, plus the regenerated golden vectors (decision D9).
- **Tag rule.** A scalar and a stdlib container have a **constant** tag (`int`, `str`, `list`,
  `OrderedDict`, `date`, `UUID`, `Decimal`, `Path`, `numpy.ndarray`, ...), so a subclass hashes like
  its base and a library moving a class to another module does not change a hash. A user-defined
  type (dataclass, model, enum, custom object) uses `module.qualname`, because the class **is** its
  identity. A tag is never derived from `type(data)` anywhere else (F8). A named tuple adds its class
  name and field names to the tag of `tuple` (so it differs from a plain tuple and from another
  named tuple).
- **Big payloads** are reduced with `digest()` by the hasher (arrays, `bytes`), so the preimage of a
  leaf is a few hundred bytes at most.

### 3.5 Registry and dispatch

`HasherRegistry(TypeRegistry[Hasher])` keeps its role: map types to hashers and find the most
specific one along the MRO, then the fallbacks (registration order), then `object` if one is
registered (the default registry has none: an unsupported type fails loudly). It keeps
`find_hasher` and `has_hasher` and gets **one** more rule, in `find_hasher` (F9):

> A subclass of `Enum` resolves to the hasher of `Enum` unless a hasher is registered for the enum
> class itself (or for a parent that is not a builtin base of the mixin).

`StringHasher` and `ReprHasher` lose their `isinstance(data, Enum)` branches, and
`HasherRegistry.hash` is `digest_of(...).hex()[:length]` with the validation of `length`. The class
is then about 120 lines instead of 239. The registry has no knowledge of cycles, placeholders or the
recursion limit.

`get_default_registry()` is the singleton with the built-in hashers, the entries of the plugin group
`coola.hashing.hashers` and the lazy groups; `get_stable_registry()` is the same without the plugins
and is the registry of `generate_stable_content_id` and `generate_stable_uuid5` (unchanged, 22.4).
`register_hashers(mapping, *, exist_ok=False)` is keyword-only for `exist_ok`.

### 3.6 Library hashers

The hashers of the optional libraries lose their registry plumbing (`_hash_values(values, registry,
length, ignore_unhashable)` is called twenty times in `pandas.py`). They return a `Node` whose
children are plain values; the engine hashes each one with the hasher of its type:

| Library | `encode(data)`                                                                                                                                                                                                                                 |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| NumPy   | numeric dtype: leaf `tagged("numpy.ndarray", f"{dtype}\0{shape}\0" + digest(canonical bytes))` (masked arrays add the digest of the mask); `object` dtype: `Node("numpy.ndarray", [dtype, shape, items])`                                      |
| torch   | leaf as NumPy with the tag `torch.Tensor`, the dense CPU values, no device and no `requires_grad`                                                                                                                                              |
| JAX     | leaf as NumPy with the tag `jax.Array`                                                                                                                                                                                                         |
| pandas  | `Node("pandas.DataFrame", [columns key, index key, dtype and values of each column])`; a key is `(names, dtype, values)`; a categorical dtype adds its categories and `ordered`; a missing value is `None` (22.1); a `NaN` name is `None` (F7) |
| polars  | `Node("polars.DataFrame", [name, str(dtype), values, ...])` per column; numeric columns without nulls pass the NumPy array                                                                                                                     |
| pyarrow | the fixed-width types as NumPy leaves with the digest of the validity bitmap; the other types as a `Node` over `to_pylist()`                                                                                                                   |
| xarray  | `Node("xarray.DataArray", [name, dims, values, coords, attrs])` (the same three shapes as today)                                                                                                                                               |

The semantics of every library hasher are unchanged (the rules of the guide: dtype, shape, values,
no layout, `-0.0` and NaN canonical). Only the way the parts are combined changes. The helpers that
three modules share (`canonicalize_floats`, `hash_array_content`) keep their place in
`coola.hashing.numpy`.

### 3.7 Custom objects: `__hash_content__`

```python
class SupportsHashContent(Protocol):
    def __hash_content__(self) -> object: ...
```

A class that cannot be a dataclass says **what its content is** and the engine hashes it (with the
tag `custom:module.Class`, so two classes with the same content differ):

```python
class Point:
    def __hash_content__(self):
        return (self.x, self.y)
```

This replaces `SupportsHash.hash(registry, length, ignore_unhashable)`: the object no longer
recurses, so it cannot bypass the cycle check, nobody forwards `length`, and the fallback predicate
is `callable(getattr(t, "__hash_content__", None))` with no signature inspection (F10). The dunder
name cannot collide with a method of a library (`polars.Series.hash`). A dataclass keeps the
dataclass hasher even if it defines `__hash_content__` (type > fallback order, then registration
order of the fallbacks), as it did for `hash()`.

### 3.8 Secrets and pydantic

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
`unwrap_secrets`, `hash_pydantic_model`, the `on_model` callback, the second thread-local and the
`PLC0415` import of `interface` disappear (F1, F11). The default registry registers
`SecretHasher("error")` for `SecretStr` and `SecretBytes` in the pydantic lazy group. To reveal or
exclude, a caller builds a registry:

```python
registry = get_default_registry().copy()
registry.register_many(get_secret_hashers("reveal"), exist_ok=True)  # SecretStr and SecretBytes
hash_object(model, registry=registry)
```

Behaviour changes, documented: a secret in a dict or in a dataclass is now governed by the same
policy as in a model (before: an error or a `TypeNotRegisteredError`); `"exclude"` no longer
**removes** the key or the item, it replaces the value by a constant (the position of a secret is
part of the schema of the class, so it is not information; the value still is never hashed).

### 3.9 Errors and options

| Situation                                   | Behavior                                                                                                         |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `length` is not an even `int` from 2 to 128 | `ValueError` before any traversal; a `bool` or a `float` is rejected with the same message (F15)                 |
| No hasher for a type                        | `TypeNotRegisteredError` (a `KeyError`) with the remedies, or a lossy placeholder and one `UserWarning` per type |
| A container that contains itself            | `ValueError("Cannot hash a self-referencing <type>.")`                                                           |
| Nesting of any depth                        | works; the only limit is memory (no `RecursionError`, no `ValueError`)                                           |
| A secret with `on_secret="error"`           | `ValueError`, without leaking the value                                                                          |
| An exception raised by a hasher             | propagates unchanged                                                                                             |

## 4. Public API after the change

```python
from coola.hashing import (
    hash_object,  # (data, *, registry=None, length=64, ignore_unhashable=False)
    HasherRegistry,
    get_default_registry,
    get_default_hasher_registry,  # alias, unchanged
    get_stable_registry,
    register_hashers,  # (mapping, *, exist_ok=False)
    Hasher,  # protocol: encode() only
    BaseHasher,
    Node,
    tagged,
    digest,
    SupportsHashContent,
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
    # lazy, as today
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

`hash_object(data, *, registry=None, length=64, ignore_unhashable=False)` has no positional option:
the deprecated form and its two overloads are deleted. `HasherRegistry.hash(data, *, length=64,
ignore_unhashable=False)` stays as the registry-level entry point (it is what the guide and the
doctests use) and calls the same engine. The decimal, fraction and range hashers stay in
`coola.hashing.canonical` (public but not re-exported, as today).

## 5. Dependents

| Package                                       | Change                                                                                                                                                                                                                                    |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `identifier.generate_stable_content_id`       | No change of code (`hash_object(data, registry=get_stable_registry(), length=...)`). **Every identifier changes** with the format (section 3.4): the release notes and the migration page say so. The tests of the module use no literal. |
| `identifier.generate_stable_uuid5`            | Same: it hashes at `length=128`, which is now the whole digest. No code change.                                                                                                                                                           |
| `equality`                                    | None. The property "equal implies same hash" is tested from `tests/properties` for the rows of the support matrix (14.4): the `known gap` hash rows that this design closes (`MappingProxyType`, `compare=False`) are removed.            |
| `utils.recursion.raise_recursion_limit_error` | Not used by `hashing` any more (the docstring example names "hashing"; change it to another action).                                                                                                                                      |
| User code with a custom hasher                | Rewritten on `encode` (migration page, section 7 step 8). A hasher of `SupportsHash` becomes `__hash_content__`.                                                                                                                          |
| Plugins of `coola.hashing.hashers`            | Their hashers must implement `encode`; the group keeps its name.                                                                                                                                                                          |

## 6. Evidence: a prototype of the engine

A prototype of the loop of section 3.2 (explicit stack, per-call type cache, the memo, `encode`
functions for `int`, `float`, `bool`, `None`, `str`, `bytes`, `list`, `tuple`, `dict`, `set` and
`frozenset`, BLAKE2b with `person`) was run against `main`. It is not part of the repository.
**[verified]** Times are the best of five runs.

| Input                                        | `main`                        | Prototype    | Prototype with inlined leaves |
| -------------------------------------------- | ----------------------------- | ------------ | ----------------------------- |
| int                                          | 1 µs                          | 1 µs         |                               |
| small dict (3 keys, nested list and dict)    | 13 µs                         | 10 µs        |                               |
| tree: depth 5, width 3 (about 1,000 nodes)   | 4.5 ms                        | 2.7 ms       |                               |
| list of 100,000 ints                         | 141 ms                        | 71 ms        | 34 ms                         |
| list of 100,000 `str`                        | 100 ms                        | 66 ms        | 31 ms                         |
| dict of 100,000 ints                         | 312 ms                        | 179 ms       | 83 ms                         |
| set of 100,000 ints                          | 164 ms                        | 92 ms        | 35 ms                         |
| 10,000 small dicts (an int, a list, a tuple) | 110 ms                        | 83 ms        | 63 ms                         |
| list nested 100,000 levels deep              | `ValueError` at 497           | 0.18 s       |                               |
| `x = [x, x]` nested 60 times (a DAG)         | never (2^60 paths; 4 s at 20) | 0.1 ms       |                               |
| cycle                                        | `ValueError`                  | `ValueError` | `ValueError`                  |

Reading the numbers:

- **The engine is 1.3 to 2.0 times faster** with the plain Merkle composition (a digest per child),
  because the lookup is paid once per type, the digests stay bytes (no hex round trip) and the
  per-node bookkeeping is one dictionary insert.
- **"Inlined leaves"** is a variant, not part of the design: a short leaf (at most 64 bytes) is
  appended to the preimage of its parent with a two-byte length instead of being digested on its own,
  so a flat container makes one BLAKE2b call instead of one per item. It is 4 times faster than
  `main` on flat containers. It costs a second form of child entry (a digest or an inline leaf) and a
  slightly longer preimage. It is left as an optional step behind the benchmark gate (D7).
- The prototype has not the real `encode` (a method call and a `Node` per container), so the real
  engine will be somewhat slower than the table; the gate (section 8) is "no slower than `main`" on
  every input, with the margin above.
- The semantics were checked on the prototype: the hash of a dict and of a set does not depend on
  the order of insertion; `[1, 2]`, `[2, 1]` and `(1, 2)` differ; `["ab"]` and `["a", "b"]` differ;
  `[[1], 2]` and `[1, [2]]` differ.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable and
puts the measurement and the fixes that do not depend on the new engine first.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Breaking                         |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 1    | **Benchmark (F6). Done with this document.** `tests/benchmarks/test_hashing_benchmark.py` (22 benchmarks, only `hash_object(data)`), baseline of `main` saved in `.benchmarks/` (appendix). No source change.                                                                                                                                                                                                                                                                                        | no                               |
| 2    | **Fix the documentation (F13).** The wrong docstrings and comments of `interface.py` and `registry.py`, the positional form in the guide; add the **Properties** table of section 1 at the top of the guide page and in the package docstring. Documentation only.                                                                                                                                                                                                                                   | no                               |
| 3    | **Equality-consistency and type-sensitivity (F4, F7, F8).** `DataclassHasher` hashes only `compare=True` fields; a `NaN` pandas name is `None`; register `MappingProxyType`; register `UserString`; a `memoryview` gets its own tag; give `StrHasher` and `ReprHasher` the same tag rule. One regression test per defect, checked to fail without the fix (`git stash push src`).                                                                                                                    | yes (hash values)                |
| 4    | **Engine and `encode` contract (F1 to F3, F5, F6, F9, F14, F15).** Add `engine.py`, `Node`, `tagged`, `digest`; change `Hasher`/`BaseHasher` to `encode`; port the built-in hashers and the library hashers; the registry delegates to the engine and gets the enum rule; delete the thread-local state, the recursion translation, the `ARG002` noqa and the leaf functions' `length` plumbing; validate `length` once. Rewrite the unit tests of each hasher on `encode`. Benchmark gate.          | yes (custom hashers, all hashes) |
| 5    | **Custom objects (F10).** `SupportsHashContent`, `ContentHasher` and the fallback on `__hash_content__`; delete `SupportsHash`, `HashableHasher` and `_has_hash_method`.                                                                                                                                                                                                                                                                                                                             | yes                              |
| 6    | **Secrets and pydantic (F11).** `SecretHasher`, `get_secret_hashers`, the lazy registration; delete `unwrap_secrets`, `hash_pydantic_model` and the second thread-local.                                                                                                                                                                                                                                                                                                                             | yes                              |
| 7    | **Delete the dead and deprecated code (F12).** The positional form of `hash_object` and its overloads, the positional `exist_ok`, `hash_string`, `hash_bytes`, `hash_tagged`, `hash_enum`, `hash_path`; update `__all__`, the guide and the tests.                                                                                                                                                                                                                                                   | yes                              |
| 8    | **Documentation.** Rewrite `uguide/hashing.md` (properties, the encoding, `__hash_content__`, secrets, the DAG and cycle notes), the `Hashing` entry of `uguide/extending.md` (the `encode` example), `dev/architecture.md`, the migration page (`migration/v1.1_to_v2.0.md`: the format change, `encode`, `__hash_content__`, the removed names, secrets), `uguide/identifier.md` (the IDs change), and the status table of `review/suggestions.md` (22.5 to 22.8, 14.4, 14.7 with the PR numbers). | no                               |
| 9    | **Layout (optional, no behaviour change).** Merge the leaf modules that are now 15 lines each (`str.py`, `repr.py`, `enum.py`, `bytes.py`, `string.py`) into `scalar.py`, and `dataclass.py` with `hashable.py` into `object.py`. Keep `sequence.py`, `mapping.py`, `set.py` apart, like `summary`. The tests move with the modules.                                                                                                                                                                 | no                               |
| 10   | **Inlined leaves (optional, decision D7).** Only if the benchmark of step 1 shows that the plain composition is too slow for a use case. Changes the format: bump `person` and regenerate the golden vectors.                                                                                                                                                                                                                                                                                        | yes (hash values)                |

Step 3 changes hash values and is independent of the engine; if the engine lands soon after, steps 3
and 4 can share the release and the golden vectors are regenerated once.

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/hashing/`); the conventions of `CLAUDE.md` apply (one
behaviour per test, literal expected values, a regression test for every bug, no local imports,
seeded data).

- **Engine** (`test_engine.py`, new):
  - the digest of a list, a tuple, a dict, a set and a nested structure (the whole digest asserted
    against a literal, one golden vector per tag);
  - a dict and a set do not depend on the order of insertion; an `OrderedDict` does;
  - `[[1], 2]`, `[1, [2]]` and `[1, 2]` differ; `["ab", "c"]` and `["a", "bc"]` differ;
  - a shared container (`[s, s]`) is encoded **once**, even if `encode` is counted by a spy hasher
    (F3), and the digest is the same as for two equal copies;
  - a cycle through a list, a dict and a tuple-in-list raises `ValueError`, whichever hasher is
    registered for the type (a custom hasher that returns its own object as a child);
  - a depth of 100,000 does not raise (regression for F2; check it fails on `main` with
    `git stash push src`);
  - an exception of a hasher propagates unchanged and leaves no state: the next call works;
  - re-entrant call (a hasher that calls `hash_object`) and a custom registry are independent;
  - the per-call cache: a hasher registered during a call is not used by this call;
  - `length`: a prefix property (`hash_object(x, length=16) == hash_object(x, length=128)[:16]`),
    the boundaries 2 and 128, `0`, `130`, an odd number, a `bool` and a `float` raise `ValueError`;
  - `ignore_unhashable`: the placeholder depends on the type only, one `UserWarning` per type per
    call (F14), a nested object and a key are replaced, `TypeNotRegisteredError` by default.
- **Each hasher** (`test_<module>.py`): `encode` asserts the whole output against a literal (the
  bytes of a leaf, the tag, the children and the flags of a node), the typical, empty, single-element
  and duplicate cases, the subclass rule (a subclass hashes like its base), the input is not modified.
  Mapping: unorderable keys, mixed key types, an `OrderedDict`, a `MappingProxyType`. Dataclass:
  `compare=False`, `init=False`, a frozen and a slotted class, a nested dataclass. `UserString`
  (F4). `memoryview` differs from `bytes` (F8). A user class named like a stdlib type does not
  collide (F8).
- **Contract test over the default registry** (`test_contract.py`, new): for one sample per
  registered type, `encode` returns `bytes` or a `Node`, twice the same value, does not modify the
  sample, and a `Node` has a non-empty `tag`, and `ordered=False` has `group >= 1`; the whole digest
  of every sample is a golden vector (replaces `test_golden.py`, regenerated once in step 4).
- **Equality-consistency** (`tests/properties/test_hashing.py`, extend): for the pairs of the
  support matrix that are equal, the hashes are equal (F7), including a dataclass with a
  `compare=False` field and a pandas `Series` with a `NaN` name; a new test replaces the missing one of 14.7.
- **Secrets** (`test_pydantic.py`): the three policies for a secret in a model, in a dict, in a list
  and in a dataclass; `"exclude"` hashes models that differ only by the secret the same; the error
  message never contains the secret; no thread-local state remains.
- **Integration** (`tests/integration/hashing/`): threads running `hash_object` on the default
  registry with a lock around the shared list (`run_threads`, docstring "Test that ..."); the stable
  identifiers end to end with the real registry; the default registry on one sample per supported
  library (NumPy, torch, JAX, pandas, polars, pyarrow, xarray, pydantic), with no patching;
  `PYTHONHASHSEED` independence (a subprocess).
- **Doctests** run as part of the docstrings; the guide examples are doctests.
- **Benchmark gate** (`tests/benchmarks/test_hashing_benchmark.py`, appendix): the new engine must not
  be slower than `main` on any input (target: at most the numbers of `main`; expected: 1.3 to 2.0
  times faster on containers). A failing gate rejects the step.
- Coverage: 100 % of `src/coola/hashing` (`--cov=coola`), no `# pragma: no cover`.

## 9. Decisions for the maintainers

**D1. One contract: `encode` returns a description, the engine hashes.** _Recommended: yes._ It is
the only way to make cycles, depth, sharing, `length` and the placeholder properties of the engine
instead of the discipline of 31 methods, and it is what `equality`, `summary` and `recursive` already
do. Alternative: keep `hash(data, registry, *, length, ignore_unhashable) -> str` and move the
parameters into a context object (`hash(data, ctx)`). It removes the `noqa` but a hasher still
recurses, so F1 to F3 stay and the engine cannot be an explicit loop.

Decision: yes

**D2. One digest size; `length` truncates the root.** _Recommended: yes._ A hasher no longer sees
`length`, the hashes of different lengths are prefixes of one another, and 64 bytes is the speed of
BLAKE2b anyway. The cost: the default `length=64` is half of a 128-character digest, and the
hashes of a given `length` change (they all change anyway, section 3.4). Alternative: the root alone
is computed with `digest_size=length // 2` (no prefix property, one more code path in the engine).

Decision: I want to be able to control the hash length. 

**D3. Memoize the shared containers.** _Recommended: yes._ Unlike `recursive_apply` (decision D3 of
`review/recursive.md`: an impure `func` would be called a different number of times), a hash is pure,
so the memo is unobservable, and it turns a DAG from exponential to linear (F3). The cost is one
dictionary entry per container and the objects kept alive until the call ends, which also makes the
`id` reuse of temporary children impossible. Alternative: no memo and a plain set of ancestors for
the cycles; simpler by three lines, 2^20 paths stay 4 s.

Decision: yes

**D4. Cycles.** _Recommended: keep `ValueError`._ `objects_are_equal` accepts a cyclic structure
and says "equal", so strictly "equal implies the same hash" is violated by a cycle. A deterministic
back-reference (the distance to the ancestor) would make a cycle hashable, but the digest of a
container in a cycle then depends on where it is reached from, which breaks the memo of D3 and the
"hash of a child is reusable" property. A cycle in a hashed configuration is a bug nine times out of
ten; the loud error is the better default. Revisit if a user needs it.

Decision: keep `ValueError`

**D5. The protocol of a custom object.** _Recommended: `__hash_content__(self) -> object`._ The
object says what its content is and does not recurse (F10); no signature inspection, no collision
with a library `hash` method. Alternative: keep `hash(registry, length, ignore_unhashable)`, which
forces the engine to support a re-entrant, parameterised call and keeps F10. Naming alternatives:
`__coola_hash__`, `hash_content`.

Decision: `__coola_hash__(self) -> object`

**D6. Secrets are types with a hasher, and `exclude` replaces instead of removing.** _Recommended:
yes._ It deletes a second traversal (`unwrap_secrets`), a second thread-local and the asymmetry
between a model and a dataclass (F11). The policy is the one of the registry, so a caller that wants
`reveal` builds a registry once. Alternative: keep the `on_secret` argument of the model hasher and
the `unwrap_secrets` traversal, and live with F11.

Decision: yes

**D7. Inline the short leaves in the preimage of their parent.** _Recommended: not now._ It is 4
times faster than `main` on flat containers (section 6) instead of 1.5 to 2 times, but it needs two
forms of child entry and a length prefix, and the plain composition already meets the gate. Do it
only when a benchmark or a user shows that hashing a million-element list is on a hot path.
Streaming the digests of an ordered node into a running `blake2b` object (memory) is the other
optimisation held back for the same reason.

Decision: not now

**D8. Remove the public leaf functions** (`hash_string`, `hash_bytes`, `hash_tagged`, `hash_enum`,
`hash_path`, `hash_pydantic_model`). _Recommended: remove._ They hash one value under a tag with a
`length`, which is `hash_object` for a scalar, and they keep `length` in the leaf code (F5). A caller
that wants a plain BLAKE2b of a string uses `hashlib`. `tagged` and `digest` stay because a custom
hasher needs them.

Decision: remove

**D9. A format version in the digest.** _Recommended: yes, `person=b"coola-hash-1"`._ It costs
nothing, it makes two formats impossible to confuse (a stored hash of an older format never equals a
new one by accident) and it turns a deliberate format change into a one-line constant plus the
golden vectors. Alternative: no version and rely on the golden vectors alone.

Decision: yes

**D10. Keep the stable registry and the plugin group.** _Recommended: keep._ The plugin group is how
a library adds its types, and `get_stable_registry()` is what keeps the identifiers independent of
the plugins (22.4). Nothing in this design changes them except the `encode` contract of the entries.

Decision: keep

**D11. Only `compare=True` fields of a dataclass.** _Recommended: yes._ It is the rule of
`objects_are_equal`, so equal dataclasses have the same hash (F7). A cache or a lazily computed
field should not make two equal objects different. A user who wants every field registers
`DataclassHasher(all_fields=True)` for the type, or a hasher of their own (the option is the one
extension of the class and is not needed otherwise: leave it out until someone asks).

Decision: yes

## 10. Not covered

- **Hashes of types that equality supports and hashing does not (14.4).** After this design, the
  next candidates are `Index`, `MultiIndex` and `Categorical` as values, `np.datetime64`,
  `np.timedelta64`, `array.array`, `dict_keys`, `dict_values`, `dict_items`, `slice`, `Ellipsis`,
  `re.Pattern`, `torch.dtype` and `torch.device`. Each is one hasher and one registration now that
  the contract is `encode`; they are separate items.
- **Functions, classes and other callables** are not hashed (a class in a configuration needs its
  own hasher). A `type` hasher by qualified name is a one-line registration if a user asks.
- **Hashing the content of a file** (`ResolvedPathHasher` hashes the path, not the file).
- **A cyclic structure** (decision D4) and **a structure that changes during the call** (no lock).
- **Streaming and parallel hashing**: the engine is a single loop; a huge array is already one
  call of BLAKE2b on a buffer.
- **Meta and float8 tensors (14.5)**: unchanged by this design; a separate item.
- **The `equality` side of 22.4**: unchanged.

## Appendix: effect on the size of the package

| Module                                                  | Lines today | Expected after  | Why                                                                                       |
| ------------------------------------------------------- | ----------- | --------------- | ----------------------------------------------------------------------------------------- |
| `interface.py`                                          | 404         | about 150       | no positional shim, no `_has_hash_method`, no signature binding                           |
| `registry.py`                                           | 239         | about 120       | no thread-local, no placeholder, no recursion translation; the enum rule stays            |
| `engine.py`                                             | 0           | about 90        | new (the loop of section 3.2 with its docstring)                                          |
| `base.py`                                               | 101         | about 110       | `Node`, `tagged`, `digest`, `encode` replace `hash` (docstrings are longer than the code) |
| `string.py`, `bytes.py`, `str.py`, `repr.py`, `enum.py` | 564         | about 220       | no leaf functions, no `length`, no `ARG002`, one enum rule                                |
| `sequence.py`, `mapping.py`, `set.py`                   | 290         | about 180       | `Node(...)` instead of a loop over `registry.hash`                                        |
| `dataclass.py`, `hashable.py`                           | 199         | about 100       | `__hash_content__`, no signature inspection                                               |
| `pydantic.py`                                           | 350         | about 110       | no `unwrap_secrets`, no thread-local, no `hash_pydantic_model`                            |
| `datetime.py`, `path.py`, `canonical.py`                | 410         | about 330       | no `registry`/`length`/`ignore_unhashable`, no `hash_path`                                |
| `numpy.py`, `torch.py`, `jax.py`                        | 358         | about 300       | leaf encodings, no parameters                                                             |
| `pandas.py`, `polars.py`, `pyarrow.py`, `xarray.py`     | 653         | about 480       | no registry plumbing; the children are plain values                                       |
| `__init__.py`                                           | 118         | about 100       | the removed names                                                                         |
| **Total**                                               | **3,686**   | **about 2,300** |                                                                                           |

The line counts are estimates from the prototype and from the code to delete; the gate is the
behaviour (sections 1 and 8), not the size.

## Appendix: benchmark of the old implementation

Baseline of `tests/benchmarks/test_hashing_benchmark.py` on the current implementation, to measure the
progress of the steps of section 7. The benchmarks only use `hash_object(data)` with the default
registry, so they run unchanged on the new design (no internals of the package). Run them with
`inv test.benchmark` (or `pytest tests/benchmarks/test_hashing_benchmark.py --benchmark-only
--benchmark-save=<name>`, then `--benchmark-compare=<name>`).

Machine: Apple M1, macOS (Darwin), Python 3.14.8, `pytest-benchmark` 5.3. One run, nothing else heavy
running, `main` at `314682c6` (`src/coola/hashing` identical to `d2e60e6fc`). The raw results are
saved in `.benchmarks/Darwin-CPython-3.14-64bit/0009_hashing_old_314682c6.json`. Save the run of the
new engine next to it and compare with `pytest --benchmark-compare=0009 --benchmark-only`. The
absolute numbers depend on the machine; read the speed-ups (old time / new time: above 1 is faster).

| Benchmark (`test_hash_object_<name>_benchmark`) | Input                                                                                      | Old median | Old min   | Gate (new median, at most) | Expected (section 6) |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------- | --------- | -------------------------- | -------------------- |
| `scalar`                                        | `1`                                                                                        | 1.6 µs     | 1.5 µs    | 1.6 µs                     | about the same       |
| `empty_list`                                    | `[]`                                                                                       | 1.2 µs     | 1.0 µs    | 1.2 µs                     | about the same       |
| `small_dict`                                    | `{"a": 1, "b": [2, 3], "c": {"d": 4}}`                                                     | 13.8 µs    | 13.4 µs   | 13.8 µs                    | 1.3x                 |
| `dataclass`                                     | a dataclass of a `str`, a `float` and a list of two `str`                                  | 12.3 µs    | 11.7 µs   | 12.3 µs                    | 1.3x                 |
| `deep_list[10]`                                 | list nested 10 times                                                                       | 11.7 µs    | 11.4 µs   | 11.7 µs                    | 1.2x                 |
| `deep_list[100]`                                | list nested 100 times                                                                      | 110.0 µs   | 108.0 µs  | 110.0 µs                   | 1.2x                 |
| `deep_list[300]`                                | list nested 300 times                                                                      | 327.6 µs   | 321.2 µs  | 327.6 µs                   | 1.2x                 |
| `list_of_numpy_arrays`                          | 1,000 arrays of 10 `float64`                                                               | 3.85 ms    | 3.74 ms   | 3.85 ms                    | about the same       |
| `numpy_array`                                   | 1,000,000 `float64`                                                                        | 8.66 ms    | 8.59 ms   | 8.66 ms                    | about the same       |
| `bytes`                                         | 10 MB of `bytes`                                                                           | 9.76 ms    | 9.68 ms   | 9.76 ms                    | about the same       |
| `tree[4-10]`                                    | depth 4, width 10 (10^4 leaves)                                                            | 17.25 ms   | 17.01 ms  | 17.25 ms                   | 1.6x                 |
| `tree[14-2]`                                    | depth 14, width 2 (2^14 leaves)                                                            | 59.34 ms   | 56.15 ms  | 59.34 ms                   | 1.6x                 |
| `tree[10-3]`                                    | depth 10, width 3 (59,049 leaves)                                                          | 145.75 ms  | 144.40 ms | 145.75 ms                  | 1.6x                 |
| `flat_list_of_strings`                          | list of 100,000 `str`                                                                      | 105.39 ms  | 104.25 ms | 105.39 ms                  | 1.5x                 |
| `list_of_small_dicts`                           | 10,000 dicts of an int, a list and a tuple                                                 | 115.52 ms  | 113.85 ms | 115.52 ms                  | 1.4x                 |
| `list_of_dataclasses`                           | 10,000 dataclasses (`str`, `float`, list of two `str`)                                     | 123.72 ms  | 121.85 ms | 123.72 ms                  | 1.4x                 |
| `flat_tuple`                                    | tuple of 100,000 ints                                                                      | 145.76 ms  | 145.13 ms | 145.76 ms                  | 2.0x                 |
| `flat_list`                                     | list of 100,000 ints                                                                       | 146.44 ms  | 145.64 ms | 146.44 ms                  | 2.0x                 |
| `mixed_containers`                              | list, tuple, `deque`, set, `frozenset`, `OrderedDict`, named tuples and `str`, 10,000 each | 160.52 ms  | 158.84 ms | 160.52 ms                  | 1.5x                 |
| `flat_list_of_floats`                           | list of 100,000 `float`                                                                    | 163.57 ms  | 159.73 ms | 163.57 ms                  | 2.0x                 |
| `flat_set`                                      | set of 100,000 ints                                                                        | 172.21 ms  | 170.83 ms | 172.21 ms                  | 1.8x                 |
| `flat_dict`                                     | dict of 100,000 ints                                                                       | 324.46 ms  | 322.43 ms | 324.46 ms                  | 1.7x                 |

Reading the numbers:

- **The cost is per node: about 1.4 µs per item of a list or a tuple, 1.7 µs per item of a set and
  3.2 µs per item of a dict** (a key and a value). Flat containers of 100,000 items take 145 to 325 ms.
  The deep list costs 1.1 µs per level, linear and recursive.
- **The array and `bytes` benchmarks measure BLAKE2b and the canonicalisation**, not the engine
  (10 MB in 9.8 ms, 8 MB of `float64` in 8.7 ms), so the new engine should not change them; they are
  the guard that the `digest(raw)` reduction of section 3.1 does not copy the buffer.
- **"Expected" is the ratio of the prototype of section 6** on the nearest input, not a promise: the
  real `encode` is a method call and a `Node` per container, so the real engine will be somewhat
  slower than the prototype. The gate is the "at most the old median" column.
- A list nested 500 levels deep raises `ValueError` on this implementation (F2), so the benchmark
  stops at 300. Add deeper cases (for example 10,000 and 100,000 levels), and a DAG (`x = [x, x]`
  nested 20 times, 4.2 s on this implementation), once the explicit stack and the memo have landed:
  they cannot run on the old design.

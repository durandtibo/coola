# Hashing: target design (`coola.hashing`)

This document describes the **ideal target state** of `coola.hashing`, in the form of properties and per-type rules. A short gap analysis against the current code (`src/coola/hashing/`) is at the end. The package is `coola.hashing`; there is no `coola.hasher`.

## 1. Purpose and non-goals

- Hashes are **deterministic content fingerprints** for identity, caching and deduplication.
- They are **not** cryptographic guarantees. Do not use them for passwords, signatures or integrity against an attacker.
- They are **not** Python's `hash()`. They must not depend on `PYTHONHASHSEED`, object addresses or insertion order of unordered containers.

## 2. Core properties

| #   | Property                  | Statement                                                                                                                                                      |
| --- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | Determinism               | The same value gives the same hash across calls, processes, machines, OSes and Python versions (3.10+).                                                        |
| P2  | Equality-consistent       | If `coola` considers two objects equal, they hash the same. The reverse is the aim, up to cryptographic collision odds.                                        |
| P3  | Type-sensitive            | Values of semantically different types hash differently, even when their text looks the same. `1`, `1.0`, `True`, `"1"`, `b"1"` and `[1]` are all distinct.    |
| P4  | Structure-sensitive       | Nesting and boundaries matter. `[[1], 2]`, `[1, [2]]` and `[1, 2]` are all distinct. `["ab", "c"]` and `["a", "bc"]` are distinct.                             |
| P5  | Order rules               | Ordered containers are order-sensitive. Unordered containers are order-insensitive.                                                                            |
| P6  | Composable                | A container's hash is built only from the hashes of its children, through the registry. Each type can be extended independently.                               |
| P7  | Fixed width, configurable | Output is lowercase hex of `length` characters (even, 2 to 128, default 64).                                                                                   |
| P8  | No silent false equality  | An unsupported type fails loudly by default. Any opt-out placeholder must be explicit and documented.                                                          |
| P9  | Cross-library stability   | Library scalar types (NumPy, torch, and so on) hash like the equivalent builtin where the semantic value is the same. They must not depend on library version. |
| P10 | Pure                      | Hashing has no side effects. It does not mutate the input or read the filesystem, environment or clock, unless the type is explicitly about the environment.   |

### Mechanism that provides these properties

- Hash function: BLAKE2b (`digest_size = length // 2`).
- **Every** leaf and container hash is domain-separated: `H(tag + "\0" + payload)`. The tag identifies the type family, so no two families can share a preimage. No hasher is allowed to be untagged.
- Containers concatenate **fixed-width child hashes** and hash the result. That is unambiguous without delimiters, which gives P4.
- Registry dispatch is by `type(data)` along the MRO, so the most specific registered type wins.

## 3. Per-type target behaviour

### 3.1 Decisions on the questions that matter

| Question                                 | Target                                                                                                                                                        | Reason                                                                                                                                                         |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[1, 2]` vs `(1, 2)`                     | **Different hashes**                                                                                                                                          | They are different types. `coola` equality treats them as unequal, and a `tuple` is often used as an immutable key with a different meaning. P2 and P3 follow. |
| `list` vs `deque` vs other `Sequence`    | **Different**, tagged per concrete family (`list`, `tuple`, `deque`, ...). Unknown `Sequence` subclasses use the tag of the registered base they dispatch to. | Same as above.                                                                                                                                                 |
| `1` vs `1.0` vs `True`                   | **All different**                                                                                                                                             | Different types (P3). `1 == 1.0` in Python, but identity here is by type and value.                                                                            |
| `{"a": 1, "b": 2}` vs `{"b": 2, "a": 1}` | **Same**                                                                                                                                                      | Insertion order is not semantic for mappings (P5).                                                                                                             |
| `dict` vs `OrderedDict`                  | **Different**, because the types differ. `OrderedDict` equality is order-sensitive, so it must also hash in order.                                            | P2 and P3.                                                                                                                                                     |
| `set` vs `frozenset`                     | **Different**. Each is order-insensitive.                                                                                                                     | P3 and P5.                                                                                                                                                     |
| `set` vs `list` of same items            | **Different**                                                                                                                                                 | P3.                                                                                                                                                            |
| `str` vs `bytes`                         | **Different**                                                                                                                                                 | P3.                                                                                                                                                            |
| Subclass instance vs base instance       | **Same** only for scalar value types normalised by P9. Containers and objects keep their own type tag.                                                        | Avoids hiding semantic differences.                                                                                                                            |

### 3.2 Scalars

- **`str`**: tag `str`, payload is UTF-8. No Unicode normalisation: `"é"` (NFC) and `"é"` (NFD) are different strings and hash differently. Lone surrogates must raise a clear `ValueError`, not an obscure encoding error.
- **`bytes`**, **`bytearray`**: tags `bytes` and `bytearray`, payload is the raw bytes.
- **`bool`**: tag `bool`, payload `True` or `False`. Checked **before** `int`.
- **`int`**: tag `int`, payload is the decimal text of the value. Arbitrary precision. NumPy ints and other integer-like subclasses normalise to this.
- **`float`**: tag `float`, payload is `repr`. It round-trips exactly.
  - `0.0` and `-0.0` are **different**, since they are different bit patterns.
  - All NaNs hash the same, because `nan` is a single value even though `nan != nan`.
  - `inf` and `-inf` hash by their repr.
- **`complex`**: tag `complex`, payload is `repr`. The same rules apply to each component.
- **`None`**: tag `NoneType`, with a fixed payload.
- **`Enum`**: tag `Enum`, payload is qualified class name plus member name. This is stable and does not depend on the value's hash. `IntEnum` is **not** collapsed into `int`, since it is a different type.

### 3.3 Ordered containers

- **`list`, `tuple`, `deque`, other registered sequences**: the tag is the concrete family. Payload is the child hashes in order. Empty containers give a valid hash that differs from `None`, `""` and the other empties.
- **`namedtuple`**: tag includes the class name plus the field names, so `Point(1, 2)` differs from `(1, 2)`.
- Strings and bytes are **never** iterated as sequences.

### 3.4 Unordered containers

- **`dict`, other `Mapping`**: tag `dict`, `OrderedDict`, and so on.
  - Each item is `key_hash + value_hash`. Items are **sorted by the key hash**, not by the key. Keys need not be orderable, and keys of mixed types work.
  - `OrderedDict` is the exception: it keeps insertion order, because its equality does.
  - Duplicate hashed keys are possible only when two distinct keys hash equal, for example `1` and `1.0` are distinct by P3 so they do not collide.
- **`set`, `frozenset`**: tag `set` and `frozenset`. Payload is the **sorted** child hashes. A set hasher must exist in the default registry.

### 3.5 Date and time

- **`date`, `datetime`, `time`, `timedelta`, `timezone`**: tag is the concrete class. Payload is the ISO 8601 text, or the exact integer fields for `timedelta`.
- Aware datetimes keep their offset in the payload. Equal instants in different zones are **different** values. Callers who want instant-equality should convert to UTC first. Naive and aware datetimes differ.
- `datetime` is a subclass of `date` but is dispatched to its own tag, so `date(2024,1,1)` and `datetime(2024,1,1)` differ.

### 3.6 Paths

Two clearly separate behaviours, and the default must be pure (P10):

- **`PurePath` / `Path` (default)**: tag `Path`, payload is `as_posix()`, with **no** filesystem access. Hashes stay portable across machines and cwd. Relative and absolute spellings of the same location therefore differ.
- **Opt-in resolver**: a separate `ResolvedPathHasher` calls `resolve()` for callers who want identity of the real location. It documents that the result depends on the filesystem and cwd.
- **File content hashing** (hash the bytes of the file) is a different operation and stays a separate function, not a hasher on `Path`.

### 3.7 Arbitrary objects, models and dataclasses

- **Objects with their own hash** (`SupportsHash`): tag `custom`, with the class qualified name. The object delegates to the registry for its nested data.
- **Pydantic models**:
  - Tag includes the **qualified model class name**, so two different models with the same values differ.
  - Fields go **through the registry**, so they obey every rule above (set ordering, type-sensitivity, NumPy normalisation). The JSON round-trip with `default=str` is not used.
  - Secrets follow `on_secret`: `"error"` (the default), `"reveal"`, `"exclude"`. A secret value must never end up in an error message or a repr.
- **Dataclasses**: tag is the qualified class name plus fields hashed through the registry, if registered.
- **Unregistered types**: raise `KeyError` by default (P8). Fallbacks that use `repr` or `str` are opt-in per type, and only for types whose text form is canonical, which excludes anything containing a memory address.

### 3.8 Unhashable placeholder

`ignore_unhashable=True` stays an explicit opt-in.

- The placeholder hash must be documented as **lossy**: any two unsupported objects of the same type are equal under it.
- It must be reported, for example with a warning or a callback, so a silent false-equality does not hide in a cache key.

## 4. Error handling

| Situation                            | Behaviour                                                                                                                  |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `length` is odd or outside 2 to 128  | `ValueError`                                                                                                               |
| No hasher for the type               | `KeyError` naming the type (or the placeholder if the user opted in)                                                       |
| Self-referencing container           | A clear `ValueError` or `RecursionError` with a message, detected through an identity set. It must not overflow the stack. |
| Non-UTF-8-encodable string           | `ValueError` naming the problem                                                                                            |
| Secret field and `on_secret="error"` | `ValueError`, without leaking the value                                                                                    |

## 5. Extension rules

For anyone adding a hasher:

1. Pick a unique, stable tag. Never reuse the tag of another family.
2. Hash only through `hash_tagged` / the registry. Never return a bare `hash_string(x)`.
3. Hash children with `registry.hash(child, length=..., ignore_unhashable=...)`. Forward both arguments.
4. Do not depend on object addresses, `id()`, `hash()`, dict or set iteration order, locale, timezone or the clock.
5. Register for the most general type where the semantics are identical, and for the exact type where they are not.
6. Add tests for P1 to P5 on the new type: stability, equality-consistency, type-sensitivity, boundaries and order.

## 6. Test matrix for the target

- Determinism: golden hashes for every tag, checked in CI, so any change of format is deliberate.
- Distinctness: `1`, `1.0`, `True`, `"1"`, `b"1"`, `[1]`, `(1,)`, `{1}`, `{1: 1}`, `None`.
- Boundaries: `["ab","c"]` vs `["a","bc"]`; `[[1],2]` vs `[1,[2]]`; `[]`, `()`, `{}`, `set()`, `""`.
- Order: `{"a":1,"b":2}` equals `{"b":2,"a":1}`; `{1,2}` equals `{2,1}`; `[1,2]` differs from `[2,1]`.
- Floats: `0.0` vs `-0.0`; `nan` stability; `np.float64(1.5)` equals `1.5`.
- Self-reference: `a = []; a.append(a)`.
- Cross-process: the same hash under different `PYTHONHASHSEED`.

## 7. Implementation status

The implementation in `src/coola/hashing/` follows this design, with these notes:

| Area | Status |
|---|---|
| Tagged `str`, `StrHasher`, `bytes` | Done. Every hasher is domain-separated. |
| One tag per sequence family, named tuples | Done (`list`, `tuple`, `deque`, `sequence`). |
| Mapping families, `OrderedDict` ordered | Done (`dict`, `OrderedDict`, `mapping`). |
| `set`, `frozenset` | Done (`SetHasher`, sorted item hashes). |
| `bytearray`, `time`, `timedelta`, `timezone`, `Enum`, dataclass | Done. `DataclassHasher` is opt-in per class, since dataclasses share no base type. |
| `Enum` mixins (`IntEnum`, `StrEnum`, `(str, Enum)`) | Done. `StringHasher` and `ReprHasher` delegate to `hash_enum`, because the MRO reaches `str`/`int` before `Enum`. |
| `Path` | Pure by default (`PathHasher`, accepts `PurePath`), `ResolvedPathHasher` is the opt-in resolver. |
| Pydantic | Class-tagged, fields go through the registry, nested models use the same `on_secret`. Registered by default when pydantic is installed. |
| Custom `SupportsHash` objects | Tagged with the qualified class name. |
| `ignore_unhashable` | Still available, now emits a `UserWarning`. |
| Self-referencing data | `ValueError` from `HasherRegistry.hash`, detected with an identity set per thread. |
| Lone surrogates in `str` | `ValueError` with a clear message. |
| Cross-library scalars (P9) | `float`/`complex`/`bool`/`int` subclasses normalise to builtins. NumPy scalars that do not subclass a builtin (e.g. `np.int64`) still need a registered hasher. |
| Golden hashes in CI | Existing unit tests pin hashes for the main types. A hash format change is a visible test diff. |

# Hashing strategy review (`coola.hashing`)

Note: the package is `coola.hashing` (there is no `coola.hasher`). Sources are in `src/coola/hashing/`.

## Overview

- Hashes are deterministic content hashes for identity and caching, not for security.
- Every hash is **BLAKE2b** with `digest_size = length // 2`, returned as lowercase hex. `length` must be even and between 2 and 128 (default 64). See `hash_string` in `string.py` and `hash_bytes` in `bytes.py`.
- `HasherRegistry.hash(data)` (`registry.py`) is the entry point. It finds a hasher by `type(data)` using MRO lookup, so the most specific registered type wins. It then calls `hasher.hash(data, registry=self, length, ignore_unhashable)`.
- Container hashers call back into the registry for each child, so nested data is hashed recursively.
- If no hasher matches, `registry.hash` raises `KeyError`. With `ignore_unhashable=True` it returns `hash_string(f"<unhashable:{type(data)!r}>")`. This also applies to nested elements.
- Most hashers prefix their input with a **type tag** through `hash_tagged(tag, value) = hash_string(f"{tag}\0{value}")`. The tag keeps values of different types from colliding on the same text. For example, the int `1` and the string `"1"` hash differently.
- Defaults are registered in `interface.py::_register_default_hashers`. Plugins can add more through the entry-point group `coola.hashing.hashers`. Users can add more with `register_hashers`.

## Per-type logic

| Type (registered key)             | Hasher           | Hashed input                                                        | Tag                                 |
| --------------------------------- | ---------------- | ------------------------------------------------------------------- | ----------------------------------- |
| `str`                             | `StringHasher`   | the string, UTF-8 encoded                                           | none                                |
| `bytes`                           | `BytesHasher`    | `b"bytes\0" + data`                                                 | `bytes`                             |
| `int`, `float`, `complex`, `bool` | `ReprHasher`     | `base.__repr__(data)`                                               | `int`, `float`, `complex` or `bool` |
| `NoneType`                        | `ReprHasher`     | `"None"`                                                            | `NoneType`                          |
| `list`, `tuple`, `Sequence`       | `SequenceHasher` | the concatenated child hashes                                       | `sequence`                          |
| `dict`, `Mapping`                 | `MappingHasher`  | the concatenated per-item `key_hash + val_hash`, sorted by key hash | `mapping`                           |
| `date`, `datetime`                | `DatetimeHasher` | `data.isoformat()`                                                  | `type(data).__qualname__`           |
| `pathlib.Path`                    | `PathHasher`     | `path.resolve().as_posix()`                                         | `Path`                              |

Not registered by default: `PydanticModelHasher`, `HashableHasher`, `StrHasher`, sets, and arbitrary objects.

### Strings (`StringHasher`)

The string is hashed directly with no tag. This is the only default hasher without one. A plain `str` therefore hashes the same as `hash_string(s)`. Because `str` is a `Sequence`, it has its own registration so it is not iterated character by character.

### Bytes (`BytesHasher`)

The input is `b"bytes\0" + data`, hashed as raw bytes. The tag is built in directly because the hasher does not go through `hash_tagged`. Hashing `b"a"` therefore differs from hashing `"a"`.

### Numbers, bool and None (`ReprHasher`)

- The hasher walks `(bool, int, float, complex)` in that order and uses the first base class that matches. It calls the **builtin** `base.__repr__(data)`, not `repr(data)`.
- This makes subclasses such as NumPy scalars hash like the builtin value, independent of library versions. For example, `np.float64(1.5)` gets the same hash as `1.5`.
- `bool` is checked first, so `True` is tagged `bool` and not `int`. `True` and `1` hash differently.
- Floats use `repr`, which round-trips exactly, so `0.1` is stable. `-0.0` differs from `0.0`. `nan` has a single repr, so it hashes consistently even though `nan != nan`.
- Anything else falls back to `hash_tagged(type.__qualname__, repr(data))`. That is how `None` is handled.

### Sequences (`SequenceHasher`)

`hash_tagged("sequence", "".join(child_hashes))`.

- Order is preserved.
- Each child hash has a fixed width (`length`), so the concatenation is unambiguous. `[ab, c]` cannot collide with `[a, bc]`.
- `list`, `tuple` and every other `Sequence` share the tag `sequence`, so **`[1, 2]` and `(1, 2)` hash the same**. This looks intentional, since the strategy compares structure, but it is a loss of type information.

### Mappings (`MappingHasher`)

- Each key and each value is hashed through the registry. The per-item string is `key_hash + val_hash`.
- Items are **sorted by key hash**, not by the key itself. This gives insertion-order independence without requiring keys to be orderable. Keys of mixed types, such as `1` and `"a"`, work.
- The result is `hash_tagged("mapping", "".join(items))`.
- `dict` and any other `Mapping` (for example `OrderedDict`) hash the same when their contents match. Insertion order is ignored.

### Dates (`DatetimeHasher`)

- The input is `isoformat()`, tagged with the concrete class name. `date(2024, 1, 1)` and `datetime(2024, 1, 1)` therefore differ, and so do `datetime` subclasses.
- Naive and aware datetimes differ because the offset is part of the ISO string.
- Equal instants in different time zones hash differently, because they are not normalised to UTC.

### Paths (`PathHasher` / `hash_path`)

- The hasher calls `Path.resolve()`, which follows symlinks and collapses `.` and `..`. It uses the POSIX form of the result and tags it `Path`.
- If `resolve` raises `OSError`, the unresolved path is used.
- The hash depends on the **filesystem and the current working directory**. This is path identity, not file content.
- Relative and absolute spellings of the same location hash the same. Hashes can differ across machines.

### Pydantic models (`PydanticModelHasher` / `hash_pydantic_model`)

Not in the default registry. Register it explicitly.

1. `model.model_dump(mode="python")`.
2. `unwrap_secrets` handles `SecretStr` and `SecretBytes` according to `on_secret`:
   - `"error"` (the default for the hasher) raises `ValueError`.
   - `"reveal"` uses the plain value.
   - `"exclude"` drops the field from its enclosing dict, list or set. A top-level secret raises.
   - Sets and frozensets become lists sorted by their JSON form.
3. `json.dumps(..., sort_keys=True, separators=(",", ":"), default=str)`.
4. `hash_bytes(b"pydantic\0" + utf8_bytes)`.

This hasher does **not** go through the registry for its fields. The model class name is not part of the hash, so two different model classes with the same field values collide.

### Other hashers (opt-in)

- `HashableHasher`: delegates to `data.hash(registry=..., length=..., ignore_unhashable=...)` for any object that satisfies the `SupportsHash` protocol. The object controls its own hash and can use the registry for nested data.
- `StrHasher`: `hash_string(str(data))`, with no tag. It is a fallback for types whose `str()` is canonical.

## Observations

1. **Tag inconsistency.** `str` and `StrHasher` are untagged, while everything else is tagged. This is harmless today. It means a custom hasher that returns an untagged `hash_string(x)` can collide with a plain `str`.
2. **list and tuple collide.** See Sequences above. If type-sensitive hashing is ever needed, the tag should come from the concrete type.
3. **Mapping and Sequence tags are in-band text.** The tag and the value are separated by `\0`. Child hashes are hex, so no ambiguity is possible. This is sound.
4. **No set or frozenset hasher.** Hashing a set raises `KeyError` unless `ignore_unhashable=True`, in which case it returns a type-only placeholder. All sets, whatever their contents, then hash the same. That is a silent false-equality risk. Pydantic handles sets, but the registry does not.
5. **`ignore_unhashable` placeholder is type-only.** Any two unhashable objects of the same type collide. This is intended and documented as a placeholder.
6. **Subclass behaviour.** `ReprHasher` normalises numeric subclasses to the builtin hash, which is good for NumPy. Other subclasses use MRO dispatch. A `str` subclass uses `StringHasher`, a `dict` subclass uses `MappingHasher`, and an `IntEnum` hashes as an `int`.
7. **Path hashing is environment dependent.** See Paths above. It is not suitable for cross-machine cache keys.
8. **No recursion guard.** A self-referencing list or dict causes `RecursionError`.
9. **Pydantic isolation.** The class name is excluded from the hash, and fields do not use the registry. Custom types inside a model fall back to `default=str`, which can embed memory addresses and make the hash unstable.

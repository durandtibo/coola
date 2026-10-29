# Architecture of `coola.io`

:book: This page describes the design of the `coola.io` package for contributors: the invariants, the contracts between the parts, and the decisions behind them. The rules that users rely on are in the [user guide](../uguide/io.md#properties); the API is in the [reference](../refs/io.md).

## Overview

The package reads and writes one value to one file, and a write either happens completely or leaves
the target as it was. The base classes own the invariants and a format only describes how to turn a
value into a file and back. `BaseLoader.load` is a template method: it converts the path (`str`,
`file://` URI or `os.PathLike`) and calls the `_load(Path)` of the format, so a format cannot forget
the conversion. `BaseFileSaver.save` writes to a hidden temporary file next to the target and
commits it with one atomic operation (`os.link` for `exist_ok=False`, `os.replace` for
`exist_ok=True`), then removes the temporary file, so a reader sees the old file or the new one and
two concurrent saves never mix. There is no lock, no lock file and no thread: the commit already
decides every race, and a caller that needs mutual exclusion around a read-modify-write uses its own
lock. `equal()`, `repr` and `str` are written once in the base classes from `_display_fields()`; a
format that has no configuration defines nothing. `TextSaver` accepts a `str` only. Optional
libraries (torch) are imported lazily. See the [user guide](../uguide/io.md#properties) for the
properties of the package.

## Design constraints

1. **Simple and easy to maintain.** The package is 1,796 lines (880 in `base.py`). The target removes
   the code that exists only to work around the design (the lock protocol, eight copies of `equal`,
   the `noqa: ARG002` that go with them, the repeated `sanitize_path` call) instead of adding code
   next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The release notes list the changes
   (`docs/docs/migration/v1.1_to_v2.0.md`).
3. **No new dependency.** `coola` has zero hard dependencies; the package uses the standard library
   only (torch stays optional and lazy).
4. **No slower than before.** `save_json` costs 773 µs on the previous version with the lock, 521 µs without it. The benchmark gate must hold.
5. **Consistent with the siblings.** Keyword-only options, `equal(other, equal_nan=False)` for the comparison of
   two configured objects (the method the package has before and the guide documents), the engine owns the invariants,
   errors name the cause, optional libraries (torch) stay lazy.
6. **Security does not regress.** `TorchLoader` stays `weights_only=True` by default, `PickleLoader`
   keeps its warning, `resolve_loader` / `resolve_saver` keep `allowed_prefixes`.
7. **Async is out of scope** (decision of the maintainers). The design must not prevent it: the sync
   core holds no lock, no thread and no global state, so a thread-based layer can be added later
   without changing it.

## Architecture in detail

```
saver.save(data, path, *, exist_ok, fsync)            loader.load(path)
        |                                                    |
        |  sanitize_path ; checks ; hidden tmp file          |  sanitize_path
        v                                                    v
self._save_file(data, tmp)    <- the format        self._load(path)      <- the format
        |
        |  fsync? ; commit (link | replace) ; remove tmp ; fsync dir?
        v
the target
```

The base classes own the invariants (paths, checks, atomicity, cleanup, durability, equality,
display). A format **only describes how to turn a value into a file and back**. This is the
same split as `recursive` (the engine owns the protections, a transformer describes the shape) and
`equality`.

### The base classes

```python
class BaseLoader(InlineDisplayMixin, ABC, Generic[T]):
    def load(self, path: str | os.PathLike[str]) -> T:
        return self._load(sanitize_path(path, resolve=False))

    @abstractmethod
    def _load(self, path: Path) -> T: ...

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        # same type and same _display_fields(), compared with objects_are_equal
        ...


class BaseSaver(InlineDisplayMixin, ABC, Generic[T]):
    @abstractmethod
    def save(self, to_save: T, path: str | os.PathLike[str], *, exist_ok: bool = False) -> None: ...

    def equal(self, other: Any, equal_nan: bool = False) -> bool: ...


class BaseFileSaver(BaseSaver[T]):
    def save(self, to_save, path, *, exist_ok=False, fsync=False) -> None: ...  #

    @abstractmethod
    def _save_file(self, to_save: T, path: Path) -> None: ...
```

- `InlineDisplayMixin` moves into the base classes, so a format defines `_display_fields()` once and
  gets `repr`, `str` and `equal()` (the fields shown are the fields compared, compared with
  `objects_are_equal` and the `equal_nan` argument). `equal` is concrete in the base class, so the
  eight overrides and their `noqa: ARG002` are deleted. `==` and `hash` stay the identity
  .
- A format that has no configuration defines nothing: `PickleLoader` is `_load` only.
- `T` is an honest type: `JsonLoader(BaseLoader[Any])`, `JsonSaver(BaseFileSaver[Any])`,
  `PickleLoader(BaseLoader[Any])`, `TorchLoader(BaseLoader[Any])`, `TextLoader(BaseLoader[str])`,
  `TextSaver(BaseFileSaver[str])`. The shortcuts instantiate the class without a subscript.
- `BaseSaver` / `BaseFileSaver` stay two classes: a saver for an object store or a
  database is a `BaseSaver` and overrides `save`. `fsync` and the temporary file only make sense for
  a file.

### The atomic commit (no lock)

`save` is the former body of `save` without `_save_lock`:

```python
def save(self, to_save, path, *, exist_ok=False, fsync=False):
    path = sanitize_path(path, resolve=False)
    if path.is_dir():
        raise IsADirectoryError(f"path ({path}) is a directory")
    if path.is_file() and not exist_ok:
        raise FileExistsError(f"path ({path}) already exists. Use `exist_ok=True` to overwrite")
    if exist_ok and path.is_symlink():
        path = path.resolve()  # overwrite the file, not the link
    _make_parent_dir(path)
    tmp = _tmp_path(path)
    try:
        self._save_file(to_save, tmp)
        if fsync:
            _fsync_file(tmp)  # O_RDWR
        if exist_ok:
            _copy_mode(path, tmp)
            tmp.replace(path)
        else:
            _link(tmp, path)  # os.link; falls back on check + replace
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    if fsync:
        _fsync_directory(path.parent)
```

Why the lock is not needed (the argument of , checked in):

| Race                                              | What decides it                                                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Two `exist_ok=False` saves create the same file   | `os.link` is atomic: one succeeds, the others get `FileExistsError`, whatever the number of processes.  |
| Two `exist_ok=True` saves overwrite the same file | `os.replace` is atomic: the file is the whole content of the last one. A lock only orders the same two. |
| A reader opens the file while it is replaced      | It opens the old inode or the new one, both complete.                                                   |
| Two saves write at the same time                  | They write different temporary files (the name has a uuid), so the data does not mix.                   |
| A save is killed                                  | The target is untouched; a hidden temporary file may stay (documented, ).                               |

A caller that needs **mutual exclusion** (a read-modify-write of one file) needs a lock around the
read and the write, which `save` could never give; it uses a lock library of its choice. The
`exist_ok=False` fallback without hard links is the one case where the lock had an effect; it is
listed under "Not guaranteed".

## Public API

```python
from coola.io import (
    BaseLoader,  # load, equal ; implement _load
    BaseSaver,  # save, equal ; implement save
    BaseFileSaver,  # save (atomic), equal ; implement _save_file
    JsonLoader,
    JsonSaver,
    PickleLoader,
    PickleSaver,
    TextLoader,
    TextSaver,
    TorchLoader,  # lazy (torch)
    TorchSaver,  # lazy (torch)
    load_json,
    save_json,
    load_pickle,
    save_pickle,
    load_text,
    save_text,
    load_torch,  # lazy
    save_torch,  # lazy
    is_loader_config,
    is_saver_config,
    resolve_loader,
    resolve_saver,
)
```

| Signature                                                                | Change                                                                                                                 |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `save_*(to_save, path, *, exist_ok=False, fsync=False, **format_kwargs)` | `lock` removed. `save_text` and `TextSaver` accept a `str` only.                                                       |
| `BaseFileSaver.save(to_save, path, *, exist_ok=False, fsync=False)`      | `lock` removed.                                                                                                        |
| `BaseLoader._load(path: Path)`                                           | New abstract method; `load` is concrete. A subclass that overrides `load` keeps working but loses the path conversion. |
| `BaseLoader.equal` / `BaseSaver.equal`                                   | Same signature, concrete in the base class: a format no longer overrides it.                                           |
| `TorchLoader(*, weights_only=True, **kwargs)`                            | `weights_only` is a named argument, shown in the repr and compared by `equal()`.                                       |
| `add_uuid_suffix`                                                        | Moves to `coola.utils.path`. `coola.io.utils` is deleted.                                                              |

Adding a format stays small. A NumPy `.npy` saver (a library `coola.io` does not import) is:

```python
class NumpySaver(BaseFileSaver["np.ndarray"]):
    def _save_file(self, to_save, path):
        with path.open("wb") as file:  # np.save adds ".npy" to a name that has no extension
            np.save(file, to_save)


class NumpyLoader(BaseLoader["np.ndarray"]):
    def _load(self, path):
        return np.load(path)
```

and gets the atomic write, `exist_ok`, `fsync`, `equal()`, `repr` from the bases.

## Dependents

| Package                                                                                 | Change                                                                                                                                                                   |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `coola.factory`                                                                         | None. `resolve_object` / `is_object_config` are called as before (the `factory` design keeps the signature used by `resolve_loader` and `resolve_saver`).                |
| `coola.utils.fork`                                                                      | `io` stops being a user of `track_fork`; `tests/integration/test_fork.py` loses the io test. The identifier generators, `LazySingleton` and the registries still use it. |
| `coola.utils.path`                                                                      | Gains `add_uuid_suffix` (with its tests moved from `tests/unit/io/test_utils.py`).                                                                                       |
| `coola.equality`                                                                        | `equal()` calls `objects_are_equal` on the fields, as `JsonSaver.equal` does before. No change in `equality`.                                                            |
| Users of `lock=`, `equal()`, `_load`-less subclasses, `add_uuid_suffix` from `coola.io` | The breaking changes of.                                                                                                                                                 |

## Decision record

**D1. Remove the save lock?** _Recommended: yes._ The atomic commit already gives the guarantees
the lock claims; the lock adds a thread, a lock file, a polling
loop, 250 µs and a deadlock. Consequences,
all listed in "Not guaranteed": two processes creating the same file at the same instant on a file
system **without hard links** can both succeed (the last one wins, no torn file); a caller who wants
mutual exclusion around a read-modify-write uses its own lock. Alternatives: (a) keep the lock as an
opt-in (`lock=True`, default off): the code and its 63 tests stay; (b) keep it for the fallback only (the link-less case): about 100 lines for a case
that affects FAT and some container volumes. If the maintainers want (b), keep `lock` out of the
public signature and take the file lock only inside `_link`'s fallback.

Decision: yes

**D2. Keep `equal()`, make it concrete?** _Recommended: yes._ The method is the documented API and
`objects_are_equal(a, b)` already works with it. Moving one implementation into the base class
(same type, then `objects_are_equal` on `_display_fields()` with `equal_nan`) deletes the eight
overrides and four `noqa` without a breaking change for callers. `==` and `hash` stay the identity, so
loaders remain hashable. Alternative: `__eq__` as in `equality` (rejected: unhashable objects and a
change of a documented method for no gain). Cost: a format whose comparison differs from its displayed
fields overrides `equal`; `TorchLoader` does not need to once `weights_only` is a named argument.

Decision: yes

**D3. How is async implemented?** _Dropped: async is out of scope._ If it comes back, the
recommendation stands: `asyncio.to_thread` of the sync method.

**D4. Async shortcuts (`aload_json`, `asave_json`, ...)?** _Dropped: async is out of scope._

**D5. Cancellation flag?** _Dropped: async is out of scope._ It is only needed by an async layer
.

**D6. `TextSaver` strict?** _Recommended: yes._ Writing `None` as the text "None" is the kind of
silent conversion the architecture guide rejects, and the class is typed `str`. The caller who wants
`str(x)` writes it, and the error says so. Alternative: keep `str()`: the typing stays dishonest.

Decision: yes

**D7. `_load` as the hook (template method)?** _Recommended: yes._ A custom loader cannot forget the
path conversion, and the base class is where a future cross-cutting step (a log line, a size
check) would go. Cost: custom subclasses rename `load` to `_load` (they keep working if they do not,
without the conversion). The saver side already works this way (`_save_file`).

Decision: yes

**D8. `add_uuid_suffix` out of `coola.io`?** _Recommended: yes, to `coola.utils.path`._ It is a path
function (`CLAUDE.md`: shared helpers belong in `coola.utils`) and only the saver uses it.

Decision: yes

**D9. Choose the format from the suffix (`load("data.json")`)?** _Recommended: not in this design._
A registry from suffix to loader is the natural next step (it would follow `coola.registry`), but it
is a new extension point and a security question: `load("x.pkl")` would unpickle by the file name.
Explicit loaders keep the format visible at the call site. If it is added, `PickleLoader` is not
registered by default.

Decision: follow recommendation

**D10. Keep `BaseSaver` and `BaseFileSaver` apart?** _Recommended: yes._ The split costs a few
lines and gives the seam for a saver that is not a local file. Merging them would put `fsync` and the temporary file on every saver.

Decision: yes

## Not covered

- **Remote and virtual file systems** (S3, GCS, `fsspec`): `path` is a local path. A saver for an
  object store is a `BaseSaver` subclass; the atomic write is the store's job. Accepting a file
  object instead of a path (so `fsspec.open` could be used) is a separate design.
- **Async (`aload`, `asave`)**: out of scope by decision of the maintainers (D3-D5). The design
  keeps the door open: after the sync core holds no lock, no thread and no global state, and a
  loader or a saver holds its configuration only, so it is safe to call from a worker thread. A layer
  can then be added without touching the core: `aload` / `asave` as `await asyncio.to_thread(self.load,
...)` in the base classes (no dependency; `aiofiles` and `anyio.Path` use threads too), `asyncio` only.
  Evidence kept from the probes on the previous version (macOS, Python 3.14): a save of 400 ms called in a coroutine
  froze a 5 ms ticker for 414 ms (6 ms with `to_thread`); `wait_for(asyncio.to_thread(save), 0.1)`
  raises `TimeoutError` and the file appears 0.4 s later, so a layer needs a cancellation flag that
  `BaseFileSaver` checks after the write and before the commit (a prototype left the target untouched);
  `to_thread(lambda: None)` costs 36 µs; 300 small saves cost 550 µs each in a blocking loop, 582 µs
  awaited one by one and 226 µs with `gather`. Not decided: async shortcuts (`asave_json`), a client
  that is async only (it would need separate `BaseAsync*` ABCs; the sync hook stays mandatory), and
  other event loops (`trio`, `anyio` call the sync methods through their own `to_thread`).
- **Streaming and large values.** One value is read or written per call, fully in memory. A
  `JsonLinesLoader` (an iterator of values) is a separate item.
- **Compression** (`.json.gz`), **checksums** and **file locking for the caller**: out of scope.
- **More formats** (`yaml`, `.npy`, `safetensors`, `bytes`): each is one `_load` and one `_save_file`
  and an optional dependency imported lazily; none is part of this design.
- **Cleanup of the leftover temporary files of killed processes**: they are hidden and have a
  recognisable name; a cleaner would need a lock to know the owner is dead, which is the protocol this
  design removes.
- **Selection by suffix**.

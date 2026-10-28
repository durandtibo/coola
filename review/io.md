# `coola.io`: target state and the changes to reach it

This document describes the design that `coola.io` should have, then the changes to make to the
current implementation (`main`, `b033254e`) to get there. The design follows the rules of the user
guide (`docs/docs/uguide/io.md`, **Properties**) and the architecture guide
(`docs/docs/dev/architecture.md`: strict contracts, lazy imports, no hidden global state), and it
stays aligned with the sibling packages (`recursive`, `factory`, `equality`: one place that owns the
invariants, a subclass that only describes its format, keyword-only options, errors that name the
cause). Its aim is a small package that is easy to maintain: **one atomic write, one template per
direction, one thin class per format**. Breaking changes are
accepted, and the future state is more important than the past one.

Evidence comes from probes and a throw-away prototype run on `main` (Python 3.14, macOS APFS; scripts
kept outside the repository). Items marked **[verified]** were reproduced; **[reading]** comes from
the code. Nothing in `src`, `tests` or `docs` was changed.

The headline findings:

- **The save lock is the largest part of the package and it protects nothing that the atomic commit
  does not already protect.** `base.py` is 880 lines and about 280 of them (a lock file, a heartbeat
  thread, stale-lock breaking, an in-process lock, a fork reset) serialize saves that the commit
  (`os.link` for `exist_ok=False`, `os.replace` for `exist_ok=True`) already makes safe. A prototype
  without any lock gave exactly one winner in 30 rounds of 8 racing processes and an intact file
  after 240 overwrites (section 6). The lock costs 250 µs per save, and blocks a thread while it waits
  (section 2, F1 and F2).
- **The contract of a loader is easy to break**: each `load` must remember to sanitize the path, and
  the example of the user guide does not (F5). The template method moves that into the base class.

## Constraints

1. **Simple and easy to maintain.** The package is 1,796 lines (880 in `base.py`). The target removes
   the code that exists only to work around the design (the lock protocol, eight copies of `equal`,
   the `noqa: ARG002` that go with them, the repeated `sanitize_path` call) instead of adding code
   next to it.
2. **Major release.** Breaking changes are fine and compatibility is not a design input: no adapter,
   no alias, no deprecation period. The release notes list the changes
   (`docs/docs/migration/v1.1_to_v2.0.md`).
3. **No new dependency.** `coola` has zero hard dependencies; the package uses the standard library
   only (torch stays optional and lazy).
4. **No slower than today.** `save_json` costs 773 µs on `main` with the lock, 521 µs without it
   **[verified]**. The benchmark gate of section 8 must hold.
5. **Consistent with the siblings.** Keyword-only options, `equal(other, equal_nan=False)` for the comparison of
   two configured objects (the method the package has today and the guide documents), the engine owns the invariants,
   errors name the cause, optional libraries (torch) stay lazy.
6. **Security does not regress.** `TorchLoader` stays `weights_only=True` by default, `PickleLoader`
   keeps its warning, `resolve_loader` / `resolve_saver` keep `allowed_prefixes`.
7. **Async is out of scope** (decision of the maintainers). The design must not prevent it: the sync
   core holds no lock, no thread and no global state, so a thread-based layer can be added later
   without changing it (section 10).

## 1. Properties of the sub-package

`coola.io` reads and writes one value to one file through a small, format-agnostic interface, and a
write either happens completely or leaves the target as it was. These are the rules of the package.
Every design choice below exists to make one of them true **by construction** (in the base classes)
instead of by the discipline of each format. They replace the table of the user guide.

| Property                 | Statement                                                                                                                                                                                                                                                            | Enforced by                 |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Atomic write             | A file saver writes to a hidden temporary file next to the target and commits it with one atomic operation. A reader sees the old file or the new file, never a partial one, and an interrupted `save` leaves the target as it was.                                  | `BaseFileSaver.save`        |
| No silent overwrite      | `save` raises `FileExistsError` if the target exists, unless `exist_ok=True`. This holds when the file is created concurrently, because the commit is a hard link.                                                                                                   | `BaseFileSaver.save`        |
| No torn file             | Concurrent saves of the same path, in one process or in several, never produce a mix: the file is the complete content of one of them. Which one is unspecified. With `exist_ok=False`, exactly one wins and the others raise `FileExistsError`.                     | the atomic commit (no lock) |
| Clean failure            | If writing fails, the temporary file is removed and the original exception is raised, even if the cleanup itself fails. An existing target is left untouched.                                                                                                        | `BaseFileSaver.save`        |
| Opt-in durability        | By default the data only reaches the operating system cache. `fsync=True` flushes the file before the commit and the directory after it, so the file survives a power failure. It is slower.                                                                         | `BaseFileSaver.save`        |
| Directory safety         | A path that is a directory raises `IsADirectoryError`, a parent that is a file raises `NotADirectoryError`, missing parents are created. A symbolic link target is overwritten through the link, and the mode of an overwritten file is kept.                        | `BaseFileSaver.save`        |
| Path-like input          | Every method takes a `str` or any `os.PathLike`; `~` and `file://` URIs are expanded. A subclass receives a `Path` and cannot forget it.                                                                                                                             | the template methods        |
| Format-agnostic API      | Every loader has `load(path)` and every saver has `save(data, path, exist_ok=...)`. Formats can be swapped without changing the calling code.                                                                                                                        | the base classes            |
| Strict input             | A saver rejects the data it cannot save with a `TypeError` or the error of the library, before it touches the target. `TextSaver` accepts a `str` only; it does not call `str()` on other objects.                                                                   | each `_save_file`           |
| Comparable               | Two loaders (or two savers) are equal (`equal()`) if they have the same type and the same configuration (encoding, keyword arguments). They hold no mutable state, so one instance can be shared by threads.                                                         | `equal` of the base classes |
| Safe by default for load | `TorchLoader` uses `weights_only=True` unless told otherwise. `PickleLoader` cannot be made safe: only load pickle files that you trust.                                                                                                                             | `TorchLoader`, the docs     |
| Extensible               | A format is added by subclassing `BaseLoader` (implement `_load`) or `BaseFileSaver` (implement `_save_file`). The atomic write, `exist_ok`, `fsync`, the paths, `equal()` and `repr` come with the base class. A backend that is not a file subclasses `BaseSaver`. | the base classes            |
| Errors propagate         | The `OSError` of the file system and the exceptions of the format library (`JSONDecodeError`, `UnpicklingError`, ...) are raised unchanged. Only the cases listed above (exists, directory, parent is a file) are translated.                                        | the template methods        |
| Lazy optional libraries  | `import coola.io` imports no optional library. `TorchLoader` and `TorchSaver` import torch the first time they are used.                                                                                                                                             | `lazy_exports`              |
| Configurable             | `resolve_loader` / `resolve_saver` build an object from a `_target_` configuration, with `allowed_prefixes` for configurations that are not trusted.                                                                                                                 | `coola.factory`             |

Not guaranteed, and documented as such:

- **Round trip beyond the format.** JSON turns tuples into lists and non-string keys into strings and
  writes `NaN` as a non-standard token; a text file stores a `str`.
- **Durability without `fsync=True`**, and which of several concurrent saves wins.
- **A hidden `.<name>-<uuid><ext>` file after a crash.** A process killed (SIGKILL, power loss)
  between the write and the commit leaves its temporary file (F3). It is hidden, never taken for the
  target, and safe to delete.
- **`exist_ok=False` on a file system without hard links** (FAT/exFAT, some network and container
  volumes): the commit falls back on "check, then rename", so two processes that create the same file
  at the same instant can both succeed. The last one wins; no file is torn.
- **Replacing a file that another process has open on Windows**: `os.replace` raises
  `PermissionError`, which propagates.
- **Streaming, compression, remote storage**: one value per call, local files only (section 10).

## 2. What differs today

| #   | Finding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Status         |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| F1  | **The lock machinery is 280 lines of `base.py` and adds no safety.** `_save_lock` takes an in-process `threading.Lock` per path, then a lock file with `O_EXCL`, then starts a **heartbeat thread** that refreshes the file every 100 s. It needs `socket`, `time`, `uuid`, `weakref`, a fork reset (`track_fork`), a stale-lock protocol with a rename-and-restore dance, a `.lock` file next to every target (so the directory must be writable and a reader that globs sees it), and it is documented as unreliable on old NFS. The commit that follows is already atomic: `os.link` fails if the path exists, `os.replace` swaps whole files. Serializing the commits does not change the outcome (the last writer wins either way). 63 of the 156 tests of the two `test_base.py` files test it. Section 6 shows the same properties without it. | **[verified]** |
| F2  | **It costs time and threads.** `save_json` takes 773 µs with the lock and 521 µs with `lock=False`: +48 %, a thread started and joined per save, and a poll loop (`time.sleep(0.01)`) while another writer holds the lock, for up to 60 s. Each waiting `save` blocks a thread.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | **[verified]** |
| F3  | **The cleanup is incomplete where it matters.** A writer killed by SIGKILL after the write leaves `.k-<uuid>.txt` **and** `k.txt.lock`. The next save breaks the lock at once (the owner pid is dead), but the temporary file stays forever. The lock protocol handles the file it created for itself and not the file that holds the data. The guide does not mention either.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | **[verified]** |
| F4  | **Re-entrance deadlocks.** A `_save_file` (or a `__reduce__` called by pickle, or a `__del__`) that calls `save` on the same path in the same thread blocks forever on the non-reentrant `threading.Lock`, which has no timeout (the 60 s timeout is only for the lock file). A 3 s join showed the thread still blocked.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | **[verified]** |
| F5  | **The contract of `load` is easy to break.** `BaseLoader.load` takes `str \| PathLike`, and every subclass must call `sanitize_path`. The custom loader of the user guide (`path.read_text()`) raises `AttributeError: 'str' object has no attribute 'read_text'` for a `str` path. The same trap exists for any user loader.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | **[verified]** |
| F6  | **Out of scope: no async.** Decision of the maintainers. For the record, a save of 400 ms called in a coroutine froze a 5 ms ticker for 414 ms, and `wait_for(asyncio.to_thread(saver.save, ...), 0.1)` raises `TimeoutError` and the file appears 0.4 s later (section 10).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | **[verified]** |
| F7  | **`TextSaver` converts anything with `str()`.** `save_text(None, path)` writes the text `None`; `save_text(123, path)` writes `123`. The class is typed `BaseFileSaver[str]` and the function takes `Any`. A caller that passes the wrong object gets a valid file with the wrong content and no error.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | **[verified]** |
| F8  | **Eight copies of `equal`.** Every loader and saver implements `equal(other, equal_nan)`: same type, then compare the fields, 4 of them with `# noqa: ARG002` (`review/noqa-review.md`). The logic is the same eight times (`==` is the identity, `equal()` is the comparison). `TorchLoader.equal` carries a special case so that the default is equal to `weights_only=True`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | **[verified]** |
| F9  | **The type parameter lies.** `JsonLoader[T]`, `PickleLoader[T]` and `TorchLoader[T]` return a `T` that nothing checks, and the four functional shortcuts instantiate `JsonLoader[Any](...)`. `TextSaver` is `BaseFileSaver[str]` and receives `Any` (F7).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | **[reading]**  |
| F10 | **`add_uuid_suffix` is a path helper exported by `coola.io`.** It is only used to name the temporary file, it is documented in the guide as a feature ("Unique File Names"), and `CLAUDE.md` says shared helpers belong in `coola.utils`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | **[reading]**  |
| F11 | **`fsync` opens the temporary file read-only** (`os.open(path, os.O_RDONLY)` in `_fsync_file`). CPython's `os.fsync` on Windows calls `_commit`, which needs a descriptor opened for writing; a read-only descriptor is expected to fail there. Not reproducible on macOS. `O_RDWR` is correct on every platform.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | **[reading]**  |
| F12 | **Documentation drift.** The options `exist_ok`, `fsync` and `lock` are described in full in six docstrings; in `save_json`, `save_pickle` and `save_torch` the `**kwargs` entry comes before `lock`, against the signature; `BaseSaver.save` has no `fsync`, which is only documented on the subclass. The guide describes the `.lock` requirement but not the leftover temporary file (F3) or the text conversion (F7).                                                                                                                                                                                                                                                                                                                                                                                                                             | **[reading]**  |
| F13 | **Tests do not mirror the source.** `tests/unit/io/test_lock_option.py` and `test_str_paths.py` have no module of `src` to mirror (`CLAUDE.md`), `test_base.py` is 1,444 lines, and there is no property test and no benchmark for `io`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | **[reading]**  |

What is already right and must not regress: write to a hidden temporary file in the target directory
and commit atomically; `os.link` for the exclusive commit and the fallback when links are not
supported; `copymode` and the symbolic link handling of an overwrite; `_short_name` so that a legal
255-byte name still gives a legal temporary name; the extension kept in the temporary name (writers
choose a format from it); the cleanup that never masks the original exception; `NotADirectoryError`
for a parent that is a file; `weights_only=True`; the security warnings; str paths and `file://`
URIs; lazy exports of the torch classes; the `_display_fields` repr.

## 3. Architecture

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

### 3.1 The base classes

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
    def save(self, to_save, path, *, exist_ok=False, fsync=False) -> None: ...  # section 3.2

    @abstractmethod
    def _save_file(self, to_save: T, path: Path) -> None: ...
```

- `InlineDisplayMixin` moves into the base classes, so a format defines `_display_fields()` once and
  gets `repr`, `str` and `equal()` (the fields shown are the fields compared, compared with
  `objects_are_equal` and the `equal_nan` argument). `equal` is concrete in the base class, so the
  eight overrides and their `noqa: ARG002` are deleted (F8). `==` and `hash` stay the identity
  (decision D2).
- A format that has no configuration defines nothing: `PickleLoader` is `_load` only.
- `T` is an honest type: `JsonLoader(BaseLoader[Any])`, `JsonSaver(BaseFileSaver[Any])`,
  `PickleLoader(BaseLoader[Any])`, `TorchLoader(BaseLoader[Any])`, `TextLoader(BaseLoader[str])`,
  `TextSaver(BaseFileSaver[str])`. The shortcuts instantiate the class without a subscript (F9).
- `BaseSaver` / `BaseFileSaver` stay two classes (decision D10): a saver for an object store or a
  database is a `BaseSaver` and overrides `save`. `fsync` and the temporary file only make sense for
  a file.

### 3.2 The atomic commit (no lock)

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
            _fsync_file(tmp)  # O_RDWR (F11)
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

Why the lock is not needed (the argument of F1, checked in section 6):

| Race                                              | What decides it                                                                                         |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Two `exist_ok=False` saves create the same file   | `os.link` is atomic: one succeeds, the others get `FileExistsError`, whatever the number of processes.  |
| Two `exist_ok=True` saves overwrite the same file | `os.replace` is atomic: the file is the whole content of the last one. A lock only orders the same two. |
| A reader opens the file while it is replaced      | It opens the old inode or the new one, both complete.                                                   |
| Two saves write at the same time                  | They write different temporary files (the name has a uuid), so the data does not mix.                   |
| A save is killed                                  | The target is untouched; a hidden temporary file may stay (documented, F3).                             |

A caller that needs **mutual exclusion** (a read-modify-write of one file) needs a lock around the
read and the write, which `save` could never give; it uses a lock library of its choice. The
`exist_ok=False` fallback without hard links is the one case where the lock had an effect; it is
listed under "Not guaranteed" (decision D1).

## 4. Public API after the change

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

## 5. Dependents

| Package                                                                                 | Change                                                                                                                                                                                                                                 |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coola.factory`                                                                         | None. `resolve_object` / `is_object_config` are called as today (`review/factory.md` keeps the signature used by `resolve_loader` and `resolve_saver`).                                                                                |
| `coola.utils.fork`                                                                      | `io` stops being a user of `track_fork`; `tests/integration/test_fork.py` loses the io test. The identifier generators, `LazySingleton` and the registries still use it.                                                               |
| `coola.utils.path`                                                                      | Gains `add_uuid_suffix` (with its tests moved from `tests/unit/io/test_utils.py`).                                                                                                                                                     |
| `coola.equality`                                                                        | `equal()` calls `objects_are_equal` on the fields, as `JsonSaver.equal` does today. No change in `equality`.                                                                                                                           |
| Docs                                                                                    | `uguide/io.md` (properties, no lock, custom loader example), `refs/io.md` (`coola.io.utils` removed), `uguide/factory.md` (link only), `migration/v1.1_to_v2.0.md` (list of section 7), `dev/architecture.md` (one paragraph on `io`). |
| `review/suggestions.md`                                                                 | Status table: the `io` items with the PR numbers (CLAUDE.md).                                                                                                                                                                          |
| Users of `lock=`, `equal()`, `_load`-less subclasses, `add_uuid_suffix` from `coola.io` | The breaking changes of section 7.                                                                                                                                                                                                     |

## 6. Evidence: probes and a prototype

All numbers are from one machine (Apple M1, APFS, Python 3.14). Scripts are outside the repository.
**[verified]**

**The current behaviour.** `save_text(None)` writes `None`; the guide's custom loader fails on a
`str`; a SIGKILL after the write leaves `['.k-4510feb7….txt', 'k.txt.lock']`; a same-path save from
inside `_save_file` is still blocked after 3 s;

**The prototype of the target save** (the code of section 3.2 as one function, no lock, no thread), stressed with forked processes on one directory:

| Test                                                                                         | Result                                                          |
| -------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| 8 processes `save(exist_ok=False)` the same path (one round)                                 | 1 winner, 7 `FileExistsError`, the file parses and is complete  |
| the same, 30 rounds                                                                          | exactly 1 winner in every round                                 |
| 8 processes × 30 `save(exist_ok=True)` of a 20,000-element list to one path (240 overwrites) | all succeed, the file is complete, the directory holds one file |

**Cost** (per save of `{"a": 1}`, 300 files):

| Variant                                                    | Time   |
| ---------------------------------------------------------- | ------ |
| `save_json` on `main` (lock, `lock=True`)                  | 773 µs |
| `save_json` on `main` (`lock=False`)                       | 521 µs |
| prototype of the target                                    | 506 µs |
| plain `open` + `json.dump`                                 | 95 µs  |
| write + `os.replace` (the `exist_ok=True` commit)          | 176 µs |
| write + `os.link` + `unlink` (the `exist_ok=False` commit) | 468 µs |

The remaining 5x over a plain write is the **hard link** on APFS (about 375 µs for the link and the
unlink); `exist_ok=True` costs 176 µs. The link is what makes `exist_ok=False` safe against a
concurrent creation, so it stays (decision D1, "deliberately not changed"). On Linux file systems a
link is much cheaper; the benchmark gate is relative to `main`.

## 7. Changes to reach the target state

One branch and one pull request per step (`CLAUDE.md`). The order keeps every step releasable.

| Step | Change                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Breaking                                |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 1    | **Fix the guide (F5, F12).** Fix the custom loader example (convert the path), move the **Properties** table to the top and replace it with the table of section 1 describe the leftover temporary file, fix the order of the `Args` of the shortcuts. Documentation only.                                                                                                                                                                                                         | no                                      |
| 2    | **Honest types and strict text (F7, F9).** `BaseLoader[Any]` / `BaseFileSaver[Any]` for the JSON, pickle and torch classes, no `JsonLoader[Any](...)` in the shortcuts, `TextSaver` raises `TypeError("... expects a str, got NoneType; call str() ...")` for a non-`str` (before the temporary file is created). Regression test for `save_text(None)`.                                                                                                                           | yes (`save_text(123)` raises)           |
| 3    | **Template methods and a shared `equal` (F5, F8).** `BaseLoader.load` is concrete and calls `_load(Path)`; `InlineDisplayMixin` and a concrete `equal` move into the bases; delete the eight `equal` and their `noqa`; `TorchLoader(weights_only=True, **kwargs)`; update the custom examples of the guide and the tests of `equal` into tests of the shared `equal` and `repr`.                                                                                                   | yes (`_load`, `equal` becomes concrete) |
| 4    | **Move `add_uuid_suffix` (F10).** To `coola.utils.path` with its tests; delete `coola.io.utils`, the export, the guide section "Unique File Names" (now in `utils`), and `docs/docs/refs/io.md`'s `coola.io.utils`.                                                                                                                                                                                                                                                                | yes (import path)                       |
| 5    | **Remove the lock (F1-F4, F11).** Delete `_save_lock`, the lock file, the heartbeat thread, `_SAVE_LOCKS`, `_SaveLocksResetter`, the `track_fork` call, the `lock=` option of the six `save` functions, `time`, `socket`, `weakref`; `_fsync_file` opens with `O_RDWR`. Delete the 63 lock tests and `test_lock_option.py`; add the integration tests of section 8 (processes: one winner, intact file; no `.lock` file). Rewrite "Requirements on the directory". Benchmark gate. | yes (`lock=` raises `TypeError`)        |
| 6    | **Property tests and layout (F13).** `tests/properties/test_io.py`, fold `test_str_paths.py` into the module tests, split what remains of `test_base.py` along the sections of `base.py`.                                                                                                                                                                                                                                                                                          | no                                      |
| 7    | **Documentation.** `dev/architecture.md` paragraph (the template split, the atomic commit), the migration page (every breaking change above), and the status table of `review/suggestions.md` with the PR numbers.                                                                                                                                                                                                                                                                 | no                                      |

Steps 2 to 5 are independent of each other once step 1 is merged; steps 6 and 7 need step 5. Step 5
can ship before step 3 if the lock is the urgent item.

## 8. Tests and gates

Unit tests mirror the source tree (`tests/unit/io/`); the conventions of `CLAUDE.md` apply (one
behaviour per test, literal expected values, a regression test for every bug, no local imports, no
sleep, `*_available` fixtures for torch).

- **Base classes** (`test_base.py`):
  - `load` converts a `str`, a `Path` and a `file://` URI before `_load`, and `_load` receives a
    `Path` (regression for F5; fails on `main` with the `str` example);
  - `equal()`: same type and fields are equal, a different field or type is not, the fields are compared with
    `objects_are_equal`, a subclass is not equal to its base;
  - the whole `repr` of a configured loader and saver against a literal;
  - `save`: `exist_ok` false raises `FileExistsError` and leaves the file untouched; `exist_ok` true
    replaces; a directory, a parent that is a file; missing parents are created; a symlink is overwritten
    through the link; the mode is kept; the temporary file is hidden, has the extension of the target and
    is removed after a failure of `_save_file`, of the commit and of the cleanup (the original error
    is raised); a 255-byte name works; `fsync=True` calls the file and the directory sync (patched,
    see the integration rule below);
  - the whole directory listing after a successful and after a failed save is the target alone, or
    empty (no `.lock`, no temporary file).
- **Formats** (`test_json.py`, `test_pickle.py`, `test_text.py`, `test_torch.py`): round trip, keyword
  arguments, encoding, `equal()` between configurations, the empty value, the error of a missing file and
  of a corrupt file, `TextSaver` rejects `None`, `int` and `bytes` with a `TypeError` and does not create a
  file, `TorchLoader()` equals `TorchLoader(weights_only=True)`.
- **Integration** (`tests/integration/io/`, `run_threads`, `requires_fork`, docstrings starting with
  "Test that ..."):
  - 8 forked processes create the same file: exactly one wins, the others see `FileExistsError`, the
    content is complete (a barrier lines them up; replaces the lock tests);
  - 8 processes overwrite the same file with distinct large content: the file is one of them;
  - 8 threads do the same with the real `save`;
  - no `.lock` file is created, and a read-only target directory raises `PermissionError`;
  - `fsync=True` with the real `os.fsync` (no patching) succeeds and leaves one file;

- **Properties** (`tests/properties/test_io.py`, new, hypothesis): a text round trip for any `str`
  without lone surrogates (line endings preserved); a pickle round trip of nested builtins; a JSON
  round trip of JSON-able values without `NaN`; `save` of an object the format rejects leaves the
  directory empty.
- **Doctests**: every docstring example runs.
- **Benchmark gate** (`tests/benchmarks/test_io_benchmark.py`, added with this document, 15 tests):
  it only uses the public functions and classes, so it runs unchanged on both designs. The appendix has the numbers of `main`. The new
  `save_json_small_new_file` and `save_json_small_overwrite` must be at most the numbers of `main`
  (the lock is gone: expected about 0.6x), `load_*` the same as `main` (± noise), `save_*_1mib` and
  `save_json_large` not slower, `save_json_same_path_4_threads` no slower than
  `save_json_distinct_paths_4_threads` + 10 % (the lock no longer serializes). A failing
  gate rejects the step.
- Coverage: 100 % of `src/coola/io` (`--cov=coola`), no `# pragma: no cover`. Baseline today: 359
  passed, 2 skipped (`tests/unit/io`, `tests/integration/io`).

## 9. Decisions for the maintainers

**D1. Remove the save lock?** _Recommended: yes._ The atomic commit already gives the guarantees
the lock claims (section 3.2, evidence in section 6); the lock adds a thread, a lock file, a polling
loop, 250 µs and a deadlock (F4). Consequences,
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
recommendation stands: `asyncio.to_thread` of the sync method (section 10).

**D4. Async shortcuts (`aload_json`, `asave_json`, ...)?** _Dropped: async is out of scope._

**D5. Cancellation flag?** _Dropped: async is out of scope._ It is only needed by an async layer
(section 10).

**D6. `TextSaver` strict?** _Recommended: yes._ Writing `None` as the text "None" is the kind of
silent conversion the architecture guide rejects, and the class is typed `str`. The caller who wants
`str(x)` writes it, and the error says so. Alternative: keep `str()`: the typing stays dishonest.

Decision: yes

**D7. `_load` as the hook (template method)?** _Recommended: yes._ A custom loader cannot forget the
path conversion (F5), and the base class is where a future cross-cutting step (a log line, a size
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

## 10. Not covered

- **Remote and virtual file systems** (S3, GCS, `fsspec`): `path` is a local path. A saver for an
  object store is a `BaseSaver` subclass (D10); the atomic write is the store's job. Accepting a file
  object instead of a path (so `fsspec.open` could be used) is a separate design.
- **Async (`aload`, `asave`)**: out of scope by decision of the maintainers (D3-D5). The design
  keeps the door open: after step 5 the sync core holds no lock, no thread and no global state, and a
  loader or a saver holds its configuration only, so it is safe to call from a worker thread. A layer
  can then be added without touching the core: `aload` / `asave` as `await asyncio.to_thread(self.load,
...)` in the base classes (no dependency; `aiofiles` and `anyio.Path` use threads too), `asyncio` only.
  Evidence kept from the probes on `main` (macOS, Python 3.14): a save of 400 ms called in a coroutine
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
  (section 4) and an optional dependency imported lazily; none is part of this design.
- **Cleanup of the leftover temporary files of killed processes**: they are hidden and have a
  recognisable name; a cleaner would need a lock to know the owner is dead, which is the protocol this
  design removes (F3).
- **Selection by suffix** (D9).

## Appendix: effect on the size of the package

| Item                                     | Lines today | Expected after | Why                                                                                        |
| ---------------------------------------- | ----------- | -------------- | ------------------------------------------------------------------------------------------ |
| `base.py` code and docstrings            | 880         | about 480      | no lock protocol (about 280 lines); the template methods add about 30                      |
| `json.py`, `pickle.py`, `text.py`        | 602         | about 480      | no `equal`, no `noqa`, no `lock` option and its docstring in each shortcut                 |
| `torch.py`                               | 201         | about 170      | same, and `weights_only` is a named argument instead of `_load_kwargs`                     |
| `utils.py`                               | 52          | 0              | moved to `coola.utils.path`                                                                |
| `__init__.py`                            | 61          | about 55       | `add_uuid_suffix` removed                                                                  |
| `tests/unit/io` + `tests/integration/io` | 3,051       | about 2,000    | 63 lock tests and `test_lock_option.py` removed; `equal`, property and process tests added |
| `tests/benchmarks/test_io_benchmark.py`  | 0           | about 80       | new                                                                                        |

The line counts are estimates from the code to delete and from the prototype; the gate is the
behaviour (sections 1 and 8), not the size.

## Appendix: benchmark of the old implementation

Baseline of `tests/benchmarks/test_io_benchmark.py` on `main` (`b033254e`), to compare with the new
implementation. Run it with `inv test.benchmark` (or
`pytest tests/benchmarks/test_io_benchmark.py --benchmark-only --benchmark-save=<name>`, then
`--benchmark-compare=<name>`). The file only calls the public API (`save_*`, `load_*`, `JsonSaver`,
`JsonLoader`), so it runs unchanged on the new design.

Machine: Apple M1, macOS (APFS, temporary directory of pytest), Python 3.14.8, `pytest-benchmark`.
One run, nothing else heavy running; the disk and the threads make the multi-thread numbers noisier
(about ±10 %) than the single-thread ones. The raw result is saved in
`.benchmarks/Darwin-CPython-3.14-64bit/0017_io_old_b033254e.json` (not committed: the directory is
ignored). The speed-up of the new implementation is old time / new time: above 1 is faster. The
"New" columns are to be filled when the steps land.

| Benchmark (`test_<name>_benchmark`)  | Input                                                   | Old median | Old min  | New median | Speed-up |
| ------------------------------------ | ------------------------------------------------------- | ---------- | -------- | ---------- | -------- |
| `load_json_small`                    | `{"key1": [1, 2, 3], "key2": "abc"}`                    | 27.8 µs    | 27.1 µs  |            |          |
| `load_json_small_class`              | same, `JsonLoader().load`                               | 27.4 µs    | 25.9 µs  |            |          |
| `load_pickle_1mib`                   | `bytes` of 1 MiB                                        | 68.0 µs    | 65.0 µs  |            |          |
| `load_text_1mib`                     | `str` of 1 MiB                                          | 177.6 µs   | 155.5 µs |            |          |
| `load_json_large`                    | list of 100,000 ints (590 KB)                           | 5.49 ms    | 5.30 ms  |            |          |
| `save_text_small_overwrite`          | `"hello"`, `exist_ok=True`                              | 452.7 µs   | 402.9 µs |            |          |
| `save_json_small_overwrite`          | the small dict, `exist_ok=True`                         | 477.3 µs   | 413.6 µs |            |          |
| `save_json_small_fsync`              | the small dict, `exist_ok=True`, `fsync=True`           | 504.1 µs   | 456.5 µs |            |          |
| `save_json_small_new_file`           | the small dict, a new path each call (`exist_ok=False`) | 819.5 µs   | 701.5 µs |            |          |
| `save_text_1mib`                     | `str` of 1 MiB, `exist_ok=True`                         | 656.0 µs   | 542.9 µs |            |          |
| `save_pickle_1mib`                   | `bytes` of 1 MiB, `exist_ok=True`                       | 709.4 µs   | 560.0 µs |            |          |
| `save_json_large`                    | list of 100,000 ints, `exist_ok=True`                   | 19.11 ms   | 18.66 ms |            |          |
| `save_json_100_files`                | 100 new files, one thread                               | 84.26 ms   | 79.73 ms |            |          |
| `save_json_distinct_paths_4_threads` | 4 threads × 25 new files                                | 34.63 ms   | 31.42 ms |            |          |
| `save_json_same_path_4_threads`      | 4 threads × 25 overwrites of one path                   | 51.69 ms   | 46.12 ms |            |          |

Reading the numbers:

- **A load is cheap, a save is not.** A small load costs 27 µs; a small save costs 0.4 to 0.8 ms,
  15 to 30 times more: the temporary file, the commit and the lock. The 1 MiB loads (68 µs for
  pickle, 178 µs for text) are limited by the read; the 1 MiB saves are dominated by the fixed cost of
  the save (0.56 to 0.71 ms), not by the size.
- **The new-file save is the slowest small save** (820 µs against 477 µs for an overwrite): the
  default `exist_ok=False` commit is `os.link` + `unlink`, which costs about 375 µs on APFS against
  176 µs for `os.replace` (section 6). The design keeps the link (D1), so the expected gain of step 5
  is the lock (about 250 µs), not the whole difference.
- **The lock serializes the threads of one path.** 100 saves from 4 threads take 51.7 ms to one path
  and 34.6 ms to 4 distinct paths (+49 %), although the per-path lock only protects the commit that is
  already atomic (F1). With 100 sequential saves at 84 ms the 4 threads give 2.4x on distinct paths.
- **`fsync=True` adds only 27 µs** (504 against 477 µs) because macOS `fsync` does not force the
  data out of the drive cache (that needs `F_FULLFSYNC`). The number says nothing about Linux, where an
  fsync costs milliseconds on a disk; rerun the file there before the step 5 decision on `O_RDWR`.
- **`json.dump` dominates a large save** (19.1 ms for 590 KB against 5.5 ms to load): the I/O layer is
  below 1 ms of it. The design cannot change this number; it is in the file to show that the gain
  applies to small files and to threads.
- Add deeper cases when the new design lands if a regression is suspected (a `TorchSaver` benchmark
  needs torch and is left out to keep the file runnable with no optional library).

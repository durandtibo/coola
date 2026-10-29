# Data Loaders and Savers

:book: This page describes the `coola.io` package, which provides a small set of loaders and
savers to read and write data to files, with a consistent, format-agnostic API.

**Prerequisites:** You'll need to know a bit of Python.
For a refresher, see the [Python tutorial](https://docs.python.org/tutorial/).

## Overview

The `coola.io` package provides:

1. **Base classes** - `BaseLoader`, `BaseSaver`, and `BaseFileSaver` to define custom loaders
   and savers
2. **Format-specific loaders and savers** - JSON, pickle, text, and PyTorch
3. **Functional shortcuts** - `load_*`/`save_*` functions for one-off usage without instantiating
   a loader or saver
4. **Configuration utilities** - `resolve_loader`/`resolve_saver` to instantiate a loader or
   saver from a configuration, and `is_loader_config`/`is_saver_config` to check configurations

## Properties

`coola.io` reads and writes one value to one file through a small, format-agnostic interface, and a
write either happens completely or leaves the target as it was. The built-in loaders and savers are
designed around the following properties.

| Property                 | Statement                                                                                                                                                                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Atomic write             | A file saver writes to a hidden temporary file next to the target and commits it with one atomic operation. A reader sees the old file or the new file, never a partial one, and an interrupted `save` leaves the target as it was.              |
| No silent overwrite      | `save` raises `FileExistsError` if the target exists, unless `exist_ok=True`. This holds when the file is created concurrently, because the commit is a hard link.                                                                               |
| No torn file             | Concurrent saves of the same path, in one process or in several, never produce a mix: the file is the complete content of one of them. Which one is unspecified. With `exist_ok=False`, exactly one wins and the others raise `FileExistsError`. |
| Clean failure            | If writing fails, the temporary file is removed and the original exception is raised, even if the cleanup itself fails. An existing target is left untouched.                                                                                    |
| Opt-in durability        | By default the data only reaches the operating system cache. `fsync=True` flushes the file before the commit and the directory after it, so the file survives a power failure. It is slower.                                                     |
| Directory safety         | A path that is a directory raises `IsADirectoryError`, a parent that is a file raises `NotADirectoryError`, missing parents are created. A symbolic link target is overwritten through the link, and the mode of an overwritten file is kept.    |
| Path-like input          | Every method takes a `str` or any `os.PathLike`; `~` and `file://` URIs are expanded.                                                                                                                                                            |
| Format-agnostic API      | Every loader has `load(path)` and every saver has `save(data, path, exist_ok=...)`. Formats can be swapped without changing the calling code.                                                                                                    |
| Strict input             | A saver rejects the data it cannot save with a `TypeError` or the error of the library, before it touches the target. `TextSaver` accepts a `str` only; it does not call `str()` on other objects.                                                           |
| Comparable               | Two loaders (or two savers) are equal (`equal()`) if they have the same type and the same configuration (encoding, keyword arguments).                                                                                                           |
| Safe by default for load | `TorchLoader` uses `weights_only=True` unless told otherwise. `PickleLoader` cannot be made safe: only load pickle files that you trust.                                                                                                         |
| Extensible               | A format is added by subclassing `BaseLoader` or `BaseFileSaver`. The atomic write, `exist_ok` and `fsync` come with the base class. A backend that is not a file subclasses `BaseSaver`.                                                        |
| Errors propagate         | The `OSError` of the file system and the exceptions of the format library (`JSONDecodeError`, `UnpicklingError`, ...) are raised unchanged. Only the cases listed above (exists, directory, parent is a file) are translated.                    |
| Lazy optional libraries  | `import coola.io` imports no optional library. `TorchLoader` and `TorchSaver` import torch the first time they are used.                                                                                                                         |
| Configurable             | `resolve_loader` / `resolve_saver` build an object from a `_target_` configuration, with `allowed_prefixes` for configurations that are not trusted.                                                                                             |

Not guaranteed:

- Round trip beyond the format: JSON turns tuples into lists and non-string keys into strings and
  writes `NaN` as a non-standard token, and a text file stores a `str`.
- Durability without `fsync=True`, and which of several concurrent saves wins.
- A hidden `.<name>-<uuid><ext>` file after a crash: a process killed (SIGKILL, power loss) between
  the write and the commit leaves its temporary file. It is hidden, never taken for the target, and
  safe to delete.
- `exist_ok=False` on a file system without hard links (FAT/exFAT, some network and container
  volumes): the commit falls back on "check, then rename", so two processes that create the same file
  at the same instant can both succeed. The last one wins; no file is torn.
- Replacing a file that another process has open on Windows: `os.replace` raises `PermissionError`,
  which propagates.
- Streaming, compression and remote storage: one value per call, local files only.

## Loading and Saving Data

Each supported format has a loader class, a saver class, and two convenience functions.

### JSON

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.io import save_json, load_json
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.json")
...     save_json({"key1": [1, 2, 3], "key2": "abc"}, path)
...     data = load_json(path)
...     data
{'key1': [1, 2, 3], 'key2': 'abc'}

```

The class-based equivalent is `JsonLoader` and `JsonSaver`:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.io import JsonSaver, JsonLoader
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.json")
...     JsonSaver().save({"key1": [1, 2, 3], "key2": "abc"}, path)
...     data = JsonLoader().load(path)
...     data
{'key1': [1, 2, 3], 'key2': 'abc'}

```

`JsonSaver` forwards extra keyword arguments to `json.dump`, e.g. `indent` or `sort_keys`:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.io import JsonSaver
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.json")
...     JsonSaver(indent=2, sort_keys=True).save({"b": 1, "a": 2}, path)

```

### Pickle

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.io import save_pickle, load_pickle
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.pkl")
...     save_pickle({"key1": [1, 2, 3], "key2": "abc"}, path)
...     data = load_pickle(path)
...     data
{'key1': [1, 2, 3], 'key2': 'abc'}

```

`PickleLoader`/`PickleSaver` are the class-based equivalents, and `PickleSaver` forwards extra
keyword arguments to `pickle.dump`.

!!! danger "Only load pickle files that you trust"

    Unpickling can execute arbitrary code. Never call `load_pickle` or `PickleLoader.load` on a file
    that comes from an untrusted or unauthenticated source. For PyTorch files, `load_torch` and
    `TorchLoader` use `weights_only=True` by default, which only loads tensors and simple containers.
    Pass `weights_only=False` only for trusted files.

### Text

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.io import save_text, load_text
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.txt")
...     save_text("hello", path)
...     data = load_text(path)
...     data
'hello'

```

`TextLoader`/`TextSaver` accept an `encoding` argument (defaults to `"utf-8"`).
`save_text` and `TextSaver` accept a `str` only and raise a `TypeError` for any other object: call
`str` on the data to write its text representation.

### PyTorch

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.io import save_torch, load_torch  # doctest: +SKIP
>>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
...     path = Path(tmpdir).joinpath("data.pt")
...     save_torch({"key1": [1, 2, 3], "key2": "abc"}, path)
...     data = load_torch(path)
...     data
{'key1': [1, 2, 3], 'key2': 'abc'}

```

`TorchLoader`/`TorchSaver` forward extra keyword arguments to `torch.load`/`torch.save`
respectively, and require PyTorch to be installed.

## Overwrite Behavior

By default, every saver raises `FileExistsError` if the target path already exists. Pass
`exist_ok=True` to overwrite it:

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.io import save_text
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     path = Path(tmpdir).joinpath("data.txt")
...     save_text("hello", path)
...     save_text("world", path, exist_ok=True)

```

File savers (`BaseFileSaver` subclasses, i.e. all the built-in savers) write to a temporary file
first and then atomically move it into place, so a job interrupted while writing never leaves a
partially written file at the target path.

To build a file name that does not collide with an existing file, without `exist_ok=True`, see
`add_uuid_suffix()` in [`coola.utils`](utils.md).

### Durability

The move is atomic, but by default the data is only written to the operating system cache: if the
machine crashes right after `save`, the file can be missing or empty even though a reader never saw
a partial file. This is fine for caches and most logs. For data that must survive a power failure,
pass `fsync=True`: the data is flushed to the disk before the file is moved into place, and the
directory is flushed afterward. It is slower.

```pycon
>>> import tempfile
>>> from pathlib import Path
>>> from coola.io import save_json
>>> with tempfile.TemporaryDirectory() as tmpdir:
...     save_json({"key": "value"}, Path(tmpdir).joinpath("data.json"), fsync=True)

```

### Requirements on the directory

`save` needs write access to the _directory_ of the target path, not only to the file: the data is
written to a hidden temporary file next to the target, and then committed with one atomic operation.
`save` takes no lock and creates no lock file. A temporary file left by a process killed between the
write and the commit is not removed: it is hidden, named `.<name>-<uuid><ext>`, and safe to delete.

Concurrent saves of the same path never produce a mix. A caller that needs mutual exclusion (a
read-modify-write of one file) uses a lock of its own around the read and the write.

When `exist_ok=False`, the file is committed with a hard link, so a file created concurrently is never
overwritten. On filesystems without hard link support (FAT/exFAT, some network and container
volumes), `save` falls back to checking that the target does not exist and renaming the temporary
file.

## Comparing Loaders and Savers

Loaders and savers implement `equal()` so they can be compared, including with
[`coola.equality`](equality.md):

```pycon
>>> from coola.io import JsonLoader, TextLoader
>>> JsonLoader().equal(JsonLoader())
True
>>> JsonLoader().equal(TextLoader())
False

```

## Configuration-Based Instantiation

Loaders and savers can be instantiated from a configuration dictionary, which is convenient when
the format is only known at runtime (e.g. read from a config file):

```pycon
>>> from coola.io import resolve_loader, resolve_saver
>>> loader = resolve_loader({"_target_": "coola.io.JsonLoader"})
>>> loader
JsonLoader(encoding=None)
>>> saver = resolve_saver({"_target_": "coola.io.JsonSaver"})
>>> saver
JsonSaver(encoding='utf-8')

```

Use `is_loader_config`/`is_saver_config` to check if a configuration targets a `BaseLoader` or
`BaseSaver`:

```pycon
>>> from coola.io import is_loader_config, is_saver_config
>>> is_loader_config({"_target_": "coola.io.JsonLoader"})
True
>>> is_saver_config({"_target_": "coola.io.JsonLoader"})
False

```

## Implementing a Custom Loader or Saver

Subclass `BaseLoader` and implement `_load` to implement a custom loader. `load` is provided by
the base class: it converts a `str`, a `file://` URI or any `os.PathLike` to a `Path` and calls
`_load` with it. `equal()`, `repr` and `str` also come with the base class and use the fields
returned by `_display_fields()` (none by default); override it in a loader that has options:

```pycon
>>> from pathlib import Path
>>> from coola.io import BaseLoader
>>> class UpperTextLoader(BaseLoader[str]):
...     def _load(self, path: Path) -> str:
...         return path.read_text().upper()

```

For a saver that writes to a file, subclass `BaseFileSaver` and implement `_save_file` instead
of `save` directly - this reuses the atomic write and `exist_ok` handling described above:

```pycon
>>> from pathlib import Path
>>> from coola.io import BaseFileSaver
>>> class UpperTextSaver(BaseFileSaver[str]):
...     def _save_file(self, to_save: str, path: Path) -> None:
...         path.write_text(str(to_save).upper())

```

## What `coola.io` does not do

- **Selection of the loader by the file suffix.** There is no `load("data.json")`: the loader is
  explicit, so the format is visible at the call site. A suffix registry would also mean that
  `load("x.pkl")` unpickles a file because of its name.
- **Async.** There is no `aload` or `asave`. A save blocks the thread that calls it. In a coroutine,
  call it with `await asyncio.to_thread(saver.save, data, path)`. The loaders and savers hold their
  configuration only, so they are safe to call from a worker thread. Cancelling the wait does not
  cancel the save: the file still appears when the thread finishes.
- **Remote and virtual file systems** (S3, GCS, `fsspec`): `path` is a local path. A saver for an
  object store is a `BaseSaver` subclass that overrides `save`; the atomic write is then the job of
  the store.
- **Streaming and large values.** One value is read or written per call, fully in memory.
- **Compression** (`.json.gz`), **checksums** and **file locking for the caller.**
- **Other formats** (`yaml`, `.npy`, `safetensors`, `bytes`). Each one is a `_load` and a
  `_save_file`, and an optional dependency imported lazily. For example, a NumPy `.npy` format:

```python
class NumpySaver(BaseFileSaver["np.ndarray"]):
    def _save_file(self, to_save, path):
        with path.open("wb") as file:  # np.save adds ".npy" to a name that has no extension
            np.save(file, to_save)


class NumpyLoader(BaseLoader["np.ndarray"]):
    def _load(self, path):
        return np.load(path)
```

- **Cleaning the temporary files** left by killed processes. They are hidden and have a
  recognisable name, and a cleaner would need a lock to know that the owner is dead.

## See Also

- [`coola.equality`](equality.md): Used internally to compare loader/saver keyword arguments.
- [`coola.utils`](utils.md): Used internally to check optional-dependency availability (e.g.
  PyTorch).

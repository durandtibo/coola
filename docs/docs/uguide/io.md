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
...
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
...
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
...

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
...
{'key1': [1, 2, 3], 'key2': 'abc'}

```

`PickleLoader`/`PickleSaver` are the class-based equivalents, and `PickleSaver` forwards extra
keyword arguments to `pickle.dump`.

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
...
'hello'

```

`TextLoader`/`TextSaver` accept an `encoding` argument (defaults to `"utf-8"`). If the data
passed to `save_text` is not a string, it is converted with `str` before being written.

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
...
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
...

```

File savers (`BaseFileSaver` subclasses, i.e. all the built-in savers) write to a temporary file
first and then atomically move it into place, so a job interrupted while writing never leaves a
partially written file at the target path.

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
JsonLoader()
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

Subclass `BaseLoader` to implement a custom loader:

```pycon
>>> from pathlib import Path
>>> from typing import Any
>>> from coola.io import BaseLoader
>>> class UpperTextLoader(BaseLoader[str]):
...     def equal(self, other: Any, equal_nan: bool = False) -> bool:
...         return type(other) is type(self)
...     def load(self, path: Path) -> str:
...         return path.read_text().upper()
...

```

For a saver that writes to a file, subclass `BaseFileSaver` and implement `_save_file` instead
of `save` directly - this reuses the atomic write and `exist_ok` handling described above:

```pycon
>>> from pathlib import Path
>>> from typing import Any
>>> from coola.io import BaseFileSaver
>>> class UpperTextSaver(BaseFileSaver[str]):
...     def equal(self, other: Any, equal_nan: bool = False) -> bool:
...         return type(other) is type(self)
...     def _save_file(self, to_save: str, path: Path) -> None:
...         path.write_text(str(to_save).upper())
...

```

## See Also

- [`coola.equality`](equality.md): Used internally to compare loader/saver keyword arguments.
- [`coola.utils`](utils.md): Used internally to check optional-dependency availability (e.g.
  PyTorch).

r"""Provide loaders and savers to read and write data to files.

A loader reads one value from a file (``load(path)``) and a saver writes
one value to a file (``save(data, path, exist_ok=False)``), with the same
interface for every format: JSON, pickle, text and PyTorch. Each format
has a class (``JsonLoader``, ``JsonSaver``, ...) and a function shortcut
(``load_json``, ``save_json``, ...).

Properties of a file saver:

- Atomic write: the data is written to a temporary file that replaces the
  target in one operation, so a reader never sees a partial file and a
  failed ``save`` leaves the target untouched.
- No silent overwrite: ``save`` raises ``FileExistsError`` if the target
  exists, unless ``exist_ok=True``.
- Opt-in durability: ``fsync=True`` flushes the file and its directory.

``resolve_loader`` and ``resolve_saver`` build a loader or a saver from a
configuration. A new format is added by subclassing ``BaseLoader`` and
``BaseFileSaver``. ``TorchLoader`` and ``TorchSaver`` import PyTorch the
first time they are used.
"""

from __future__ import annotations

__all__ = [
    "BaseFileSaver",
    "BaseLoader",
    "BaseSaver",
    "JsonLoader",
    "JsonSaver",
    "PickleLoader",
    "PickleSaver",
    "TextLoader",
    "TextSaver",
    "TorchLoader",
    "TorchSaver",
    "is_loader_config",
    "is_saver_config",
    "load_json",
    "load_pickle",
    "load_text",
    "load_torch",
    "resolve_loader",
    "resolve_saver",
    "save_json",
    "save_pickle",
    "save_text",
    "save_torch",
]

from typing import TYPE_CHECKING

from coola.io.base import (
    BaseFileSaver,
    BaseLoader,
    BaseSaver,
    is_loader_config,
    is_saver_config,
    resolve_loader,
    resolve_saver,
)
from coola.io.json import JsonLoader, JsonSaver, load_json, save_json
from coola.io.pickle import PickleLoader, PickleSaver, load_pickle, save_pickle
from coola.io.text import TextLoader, TextSaver, load_text, save_text
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.io.torch import TorchLoader, TorchSaver, load_torch, save_torch


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "TorchLoader": "coola.io.torch",
    "TorchSaver": "coola.io.torch",
    "load_torch": "coola.io.torch",
    "save_torch": "coola.io.torch",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())

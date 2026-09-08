r"""Contain data loaders and savers."""

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
    "add_uuid_suffix",
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
from coola.io.torch import TorchLoader, TorchSaver, load_torch, save_torch
from coola.io.utils import add_uuid_suffix

r"""Provide deterministic hashing functions.

These are content hashes for identity and caching, not for security: do
not use them for passwords, signatures or integrity against an attacker.

Hashes are deterministic across calls, processes and machines, and
follow these rules:

- Every hash is domain-separated by a type tag, so values of different
  types never share a hash even if their text form is the same
  (``1``, ``1.0``, ``True`` and ``"1"`` are all distinct).
- Ordered containers (``list``, ``tuple``, ``OrderedDict``) are
  order-sensitive and unordered containers (``dict``, ``set``) are not,
  like in ``coola`` equality. Each container family has its own tag, so
  a ``list`` and a ``tuple`` with the same items hash differently.
- Container hashes are built only from the hashes of their children,
  resolved through a ``HasherRegistry``.
- Unsupported types raise ``KeyError`` unless ``ignore_unhashable`` is
  set.
"""

from __future__ import annotations

__all__ = [
    "BaseHasher",
    "BytesHasher",
    "DataclassHasher",
    "DatetimeHasher",
    "EnumHasher",
    "HashableHasher",
    "Hasher",
    "HasherRegistry",
    "JaxArrayHasher",
    "MappingHasher",
    "NumpyArrayHasher",
    "PandasDataFrameHasher",
    "PandasSeriesHasher",
    "PathHasher",
    "PolarsDataFrameHasher",
    "PolarsSeriesHasher",
    "PyarrowArrayHasher",
    "PyarrowTableHasher",
    "PydanticModelHasher",
    "ReprHasher",
    "ResolvedPathHasher",
    "SequenceHasher",
    "SetHasher",
    "StrHasher",
    "StringHasher",
    "SupportsHash",
    "TorchTensorHasher",
    "XarrayHasher",
    "get_default_hasher_registry",
    "get_default_registry",
    "hash_bytes",
    "hash_enum",
    "hash_object",
    "hash_path",
    "hash_pydantic_model",
    "hash_string",
    "hash_tagged",
    "register_hashers",
]

from typing import TYPE_CHECKING

from coola.hashing.base import BaseHasher, Hasher
from coola.hashing.bytes import BytesHasher, hash_bytes
from coola.hashing.dataclass import DataclassHasher
from coola.hashing.datetime import DatetimeHasher
from coola.hashing.enum import EnumHasher
from coola.hashing.hashable import HashableHasher, SupportsHash
from coola.hashing.interface import (
    get_default_hasher_registry,
    get_default_registry,
    hash_object,
    register_hashers,
)
from coola.hashing.mapping import MappingHasher
from coola.hashing.path import PathHasher, ResolvedPathHasher, hash_path
from coola.hashing.registry import HasherRegistry
from coola.hashing.repr import ReprHasher
from coola.hashing.sequence import SequenceHasher
from coola.hashing.set import SetHasher
from coola.hashing.str import StrHasher
from coola.hashing.string import StringHasher, hash_enum, hash_string, hash_tagged
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.hashing.jax import JaxArrayHasher
    from coola.hashing.numpy import NumpyArrayHasher
    from coola.hashing.pandas import PandasDataFrameHasher, PandasSeriesHasher
    from coola.hashing.polars import PolarsDataFrameHasher, PolarsSeriesHasher
    from coola.hashing.pyarrow import PyarrowArrayHasher, PyarrowTableHasher
    from coola.hashing.pydantic import PydanticModelHasher, hash_pydantic_model
    from coola.hashing.torch import TorchTensorHasher
    from coola.hashing.xarray import XarrayHasher


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "JaxArrayHasher": "coola.hashing.jax",
    "NumpyArrayHasher": "coola.hashing.numpy",
    "PandasDataFrameHasher": "coola.hashing.pandas",
    "PandasSeriesHasher": "coola.hashing.pandas",
    "PolarsDataFrameHasher": "coola.hashing.polars",
    "PolarsSeriesHasher": "coola.hashing.polars",
    "PyarrowArrayHasher": "coola.hashing.pyarrow",
    "PyarrowTableHasher": "coola.hashing.pyarrow",
    "PydanticModelHasher": "coola.hashing.pydantic",
    "TorchTensorHasher": "coola.hashing.torch",
    "XarrayHasher": "coola.hashing.xarray",
    "hash_pydantic_model": "coola.hashing.pydantic",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())

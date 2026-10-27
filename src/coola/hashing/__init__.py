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
- Unsupported types raise ``TypeNotRegisteredError`` (a ``KeyError``)
  unless ``ignore_unhashable`` is set.
- A container that contains itself raises ``ValueError``.
- The output is a lowercase hex string of ``length`` characters (an
  even number from 2 to 128, default 64).
"""

from __future__ import annotations

__all__ = [
    "BaseHasher",
    "BytesHasher",
    "ContentHasher",
    "DataclassHasher",
    "DatetimeHasher",
    "EnumHasher",
    "Hasher",
    "HasherRegistry",
    "JaxArrayHasher",
    "MappingHasher",
    "Node",
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
    "SecretHasher",
    "SequenceHasher",
    "SetHasher",
    "StrHasher",
    "StringHasher",
    "SupportsCoolaHash",
    "TorchTensorHasher",
    "XarrayHasher",
    "digest",
    "get_default_hasher_registry",
    "get_default_registry",
    "get_secret_hashers",
    "get_stable_registry",
    "hash_object",
    "register_hashers",
    "tagged",
]

from typing import TYPE_CHECKING

from coola.hashing.base import BaseHasher, Hasher, Node, digest, tagged
from coola.hashing.bytes import BytesHasher
from coola.hashing.dataclass import DataclassHasher
from coola.hashing.datetime import DatetimeHasher
from coola.hashing.enum import EnumHasher
from coola.hashing.hashable import ContentHasher, SupportsCoolaHash
from coola.hashing.interface import (
    get_default_hasher_registry,
    get_default_registry,
    get_stable_registry,
    hash_object,
    register_hashers,
)
from coola.hashing.mapping import MappingHasher
from coola.hashing.path import PathHasher, ResolvedPathHasher
from coola.hashing.registry import HasherRegistry
from coola.hashing.repr import ReprHasher
from coola.hashing.sequence import SequenceHasher
from coola.hashing.set import SetHasher
from coola.hashing.str import StrHasher
from coola.hashing.string import StringHasher
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.hashing.jax import JaxArrayHasher
    from coola.hashing.numpy import NumpyArrayHasher
    from coola.hashing.pandas import PandasDataFrameHasher, PandasSeriesHasher
    from coola.hashing.polars import PolarsDataFrameHasher, PolarsSeriesHasher
    from coola.hashing.pyarrow import PyarrowArrayHasher, PyarrowTableHasher
    from coola.hashing.pydantic import PydanticModelHasher, SecretHasher, get_secret_hashers
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
    "SecretHasher": "coola.hashing.pydantic",
    "get_secret_hashers": "coola.hashing.pydantic",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())

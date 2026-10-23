r"""Provide deterministic hashing for PyArrow objects."""

from __future__ import annotations

__all__ = ["PyarrowArrayHasher", "PyarrowTableHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged
from coola.utils.imports import is_pyarrow_available

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa

if TYPE_CHECKING:
    from coola.hashing.base import Hasher
    from coola.hashing.registry import HasherRegistry


def _hash_array(array: Any, registry: HasherRegistry, length: int, ignore_unhashable: bool) -> str:
    r"""Hash the type and the values of an array."""
    values = array.to_pylist()
    parts = [
        str(array.type),
        registry.hash(values, length=length, ignore_unhashable=ignore_unhashable),
    ]
    return hash_tagged("pyarrow.Array", "\0".join(parts), length=length)


class PyarrowArrayHasher(InlineDisplayMixin, BaseHasher["pa.Array | pa.ChunkedArray"]):
    r"""Hasher for ``pyarrow.Array`` and ``pyarrow.ChunkedArray``
    objects.

    The hash depends on the type and the values, ``null`` included. The
    chunk layout of a ``ChunkedArray`` is not part of the hash.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.hashing import PyarrowArrayHasher, get_default_registry
        >>> hasher = PyarrowArrayHasher()
        >>> hasher
        PyarrowArrayHasher()
        >>> len(hasher.hash(pa.array([1, 2, 3]), registry=get_default_registry()))
        64

        ```
    """

    def hash(
        self,
        data: pa.Array | pa.ChunkedArray,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        return _hash_array(data, registry, length, ignore_unhashable)


class PyarrowTableHasher(InlineDisplayMixin, BaseHasher["pa.Table | pa.RecordBatch"]):
    r"""Hasher for ``pyarrow.Table`` and ``pyarrow.RecordBatch``
    objects.

    The hash depends on the column names, the types and the values.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.hashing import PyarrowTableHasher, get_default_registry
        >>> hasher = PyarrowTableHasher()
        >>> hasher
        PyarrowTableHasher()
        >>> len(hasher.hash(pa.table({"a": [1, 2]}), registry=get_default_registry()))
        64

        ```
    """

    def hash(
        self,
        data: pa.Table | pa.RecordBatch,
        registry: HasherRegistry,
        *,
        length: int = 64,
        ignore_unhashable: bool = False,
    ) -> str:
        parts = []
        for name, column in zip(data.schema.names, data.columns, strict=True):
            parts.append(registry.hash(name, length=length, ignore_unhashable=ignore_unhashable))
            parts.append(_hash_array(column, registry, length, ignore_unhashable))
        return hash_tagged("pyarrow.Table", "\0".join(parts), length=length)


def get_default_hashers() -> dict[type, Hasher[Any]]:
    r"""Get the default hashers of the pyarrow objects.

    Returns:
        The hashers of the pyarrow objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the hashers of the pyarrow objects.
    """
    if not is_pyarrow_available():
        return {}
    array_hasher = PyarrowArrayHasher()
    table_hasher = PyarrowTableHasher()
    return {
        pa.Array: array_hasher,
        pa.ChunkedArray: array_hasher,
        pa.Table: table_hasher,
        pa.RecordBatch: table_hasher,
    }

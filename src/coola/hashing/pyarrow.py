r"""Provide deterministic hashing for PyArrow objects."""

from __future__ import annotations

__all__ = ["PyarrowArrayHasher", "PyarrowTableHasher"]

from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin, NoArgsDisplayMixin
from coola.hashing.base import BaseHasher
from coola.hashing.string import hash_tagged

if TYPE_CHECKING:
    import pyarrow as pa

    from coola.hashing.registry import HasherRegistry


def _hash_array(array: Any, registry: HasherRegistry, length: int, ignore_unhashable: bool) -> str:
    r"""Hash the type and the values of an array."""
    values = array.to_pylist()
    parts = [
        str(array.type),
        registry.hash(values, length=length, ignore_unhashable=ignore_unhashable),
    ]
    return hash_tagged("pyarrow.Array", "\0".join(parts), length=length)


class PyarrowArrayHasher(
    InlineDisplayMixin, NoArgsDisplayMixin, BaseHasher["pa.Array | pa.ChunkedArray"]
):
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


class PyarrowTableHasher(
    InlineDisplayMixin, NoArgsDisplayMixin, BaseHasher["pa.Table | pa.RecordBatch"]
):
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

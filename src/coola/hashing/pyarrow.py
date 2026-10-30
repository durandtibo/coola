r"""Provide deterministic hashing for PyArrow objects."""

from __future__ import annotations

__all__ = ["PyarrowArrayHasher", "PyarrowTableHasher"]

from itertools import chain
from typing import TYPE_CHECKING, Any

from coola.display import InlineDisplayMixin
from coola.hashing.base import BaseHasher, Encoding, Node, digest, tagged
from coola.hashing.numpy import canonicalize_floats
from coola.utils.imports import is_numpy_available, is_pyarrow_available

if TYPE_CHECKING or is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    from coola.utils.fallback.numpy import numpy as np

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
    import pyarrow.compute as pc
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa

    pc = pa

if TYPE_CHECKING:
    from coola.hashing.base import Hasher


def _is_fixed_width(data_type: Any) -> bool:
    r"""Indicate if the values of a type can be hashed from a NumPy array
    without going through Python objects.

    Args:
        data_type: The Arrow type.

    Returns:
        ``True`` for an integer, floating-point or boolean type,
            otherwise ``False``.
    """
    return (
        pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_boolean(data_type)
    )


def _encode_array(array: Any) -> Encoding:
    r"""Describe an array: a leaf for the fixed-width types, a ``Node``
    over ``to_pylist()`` for the others.

    Args:
        array: The Arrow array.

    Returns:
        The encoding of the array.
    """
    if not (is_numpy_available() and _is_fixed_width(array.type)):
        return Node("pyarrow.Array", [str(array.type), array.to_pylist()])
    mask = b""
    if array.null_count:
        mask = digest(
            np.ascontiguousarray(array.is_null().to_numpy(zero_copy_only=False)).tobytes()
        )
        array = pc.fill_null(array, False if pa.types.is_boolean(array.type) else 0)
    values = canonicalize_floats(array.to_numpy(zero_copy_only=False))
    return tagged(
        "pyarrow.Array",
        f"{array.type}\0".encode() + digest(np.ascontiguousarray(values).tobytes()) + mask,
    )


class PyarrowArrayHasher(InlineDisplayMixin, BaseHasher["pa.Array | pa.ChunkedArray"]):
    r"""Hasher for ``pyarrow.Array`` and ``pyarrow.ChunkedArray``
    objects.

    The hash depends on the type and the values, ``null`` included. The
    chunk layout of a ``ChunkedArray`` is not part of the hash. The
    integer, floating-point and boolean arrays are hashed from their
    NumPy form (``-0.0`` and NaN are canonicalized like in NumPy) when
    NumPy is installed, so they are as fast as NumPy arrays; the other
    types are hashed through Python objects.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.hashing import PyarrowArrayHasher
        >>> hasher = PyarrowArrayHasher()
        >>> hasher
        PyarrowArrayHasher()
        >>> hasher.encode(pa.array([1, 2, 3])).startswith(b'pyarrow.Array')
        True

        ```
    """

    def encode(self, data: pa.Array | pa.ChunkedArray) -> Encoding:
        r"""Describe an array.

        Args:
            data: The array to describe.

        Returns:
            A leaf (the type, the digest of the values and of the
            validity) for the fixed-width types, and a ``Node`` of the
            type and the Python values for the other types.
        """
        return _encode_array(data)


class PyarrowTableHasher(InlineDisplayMixin, BaseHasher["pa.Table | pa.RecordBatch"]):
    r"""Hasher for ``pyarrow.Table`` and ``pyarrow.RecordBatch``
    objects.

    The hash depends on the column names, the types and the values.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.hashing import PyarrowTableHasher
        >>> hasher = PyarrowTableHasher()
        >>> hasher
        PyarrowTableHasher()
        >>> hasher.encode(pa.table({"a": [1, 2]})).tag
        'pyarrow.Table'

        ```
    """

    def encode(self, data: pa.Table | pa.RecordBatch) -> Encoding:
        r"""Describe a table or a record batch.

        Args:
            data: The table to describe.

        Returns:
            The ``Node`` of the column names and the columns.
        """
        return Node(
            "pyarrow.Table",
            chain.from_iterable(zip(data.schema.names, data.columns, strict=True)),
        )


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

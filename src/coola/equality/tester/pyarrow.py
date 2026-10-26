r"""Implement the adapters of the pyarrow objects.

This module provides the adapters that convert a ``pyarrow.Array``,
``pyarrow.ChunkedArray``, ``pyarrow.RecordBatch`` or ``pyarrow.Table``
to a table, which is compared by the ``TableEqualityTester``.
"""

from __future__ import annotations

__all__ = ["PyarrowArrayTable", "PyarrowTableTable", "pyarrow_array", "pyarrow_table"]

from typing import TYPE_CHECKING, Any, cast

from coola.equality.tester.table import Column, TableEqualityTester
from coola.utils.imports import is_numpy_available, is_pyarrow_available

if TYPE_CHECKING or is_pyarrow_available():
    import pyarrow as pa
    import pyarrow.compute as pc
else:  # pragma: no cover
    from coola.utils.fallback.pyarrow import pyarrow as pa

    pc = pa

if TYPE_CHECKING:
    from collections.abc import Iterator

    from coola.equality.engine import Engine
    from coola.equality.tester.base import BaseEqualityTester


class PyarrowTableTable:
    r"""Adapt a ``pyarrow.Table`` or a ``pyarrow.RecordBatch`` to a table.

    The structure is the schema, without its metadata, and the number of
    rows. The columns are compared one by one, by the kind of their
    type.

    Args:
        table: The table or the record batch.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.equality.tester.pyarrow import PyarrowTableTable
        >>> table = PyarrowTableTable(pa.table({"a": [1, 2]}))
        >>> table.kind_name
        'tables'

        ```
    """

    kind_name = "tables"

    def __init__(self, table: Any) -> None:
        self._table = table

    def same_structure(self, other: PyarrowTableTable, ctx: Engine) -> bool:
        _warn_ignored(ctx, self._table.schema.types, self._table)
        return bool(
            self._table.schema.equals(other._table.schema)
            and self._table.num_rows == other._table.num_rows
        )

    def columns(self) -> Iterator[Column]:
        for name, column in zip(self._table.schema.names, self._table.columns, strict=True):
            yield _column(name, column)


class PyarrowArrayTable:
    r"""Adapt a ``pyarrow.Array`` or a ``pyarrow.ChunkedArray`` to a table
    with one column.

    The structure is the type and the length.

    Args:
        array: The array.

    Example:
        ```pycon
        >>> import pyarrow as pa
        >>> from coola.equality.tester.pyarrow import PyarrowArrayTable
        >>> table = PyarrowArrayTable(pa.array([1, 2]))
        >>> table.kind_name
        'arrays'

        ```
    """

    kind_name = "arrays"

    def __init__(self, array: Any) -> None:
        self._array = array

    def same_structure(self, other: PyarrowArrayTable, ctx: Engine) -> bool:
        _warn_ignored(ctx, [self._array.type], self._array)
        return bool(self._array.type == other._array.type and len(self._array) == len(other._array))

    def columns(self) -> Iterator[Column]:
        yield _column(None, self._array)


def pyarrow_table(table: Any) -> PyarrowTableTable:
    r"""Adapt a ``pyarrow.Table`` or a ``pyarrow.RecordBatch`` to a table."""
    return PyarrowTableTable(table)


def pyarrow_array(array: Any) -> PyarrowArrayTable:
    r"""Adapt a ``pyarrow.Array`` or a ``pyarrow.ChunkedArray`` to a table."""
    return PyarrowArrayTable(array)


def _column(name: object, array: Any) -> Column:
    r"""Convert an array to a column, by the kind of its type.

    - integers and floats: the ``null`` values must be at the same
      positions, then the values are compared with the NumPy tolerance
      rule (``atol``, ``rtol``, ``equal_nan``);
    - durations: same, on the durations (``atol`` is a number of
      seconds);
    - the other types (booleans, strings, instants, decimals, nested
      types, ...): the ``equals`` of Arrow. It never sees the tolerance
      or ``equal_nan``, a ``null`` is equal to a ``null`` and a ``NaN``
      nested in a value is not equal to a ``NaN``.

    If NumPy is not installed, ``equal_nan``, ``atol`` and ``rtol`` are
    ignored and every column is compared by Arrow (see
    ``_warn_ignored``).
    """
    kind = array.type
    if is_numpy_available() and _is_quantity(kind):
        values = pc.fill_null(array, pa.scalar(0, type=kind)).to_numpy(zero_copy_only=False)
        return Column(
            name, "duration" if pa.types.is_duration(kind) else "numeric", values, _nulls(array)
        )
    return Column(name, "exact", array, equals=_arrow_equals)


def _is_quantity(kind: Any) -> bool:
    return bool(
        pa.types.is_floating(kind) or pa.types.is_integer(kind) or pa.types.is_duration(kind)
    )


def _nulls(array: Any) -> Any:
    return array.is_null().to_numpy(zero_copy_only=False)


def _arrow_equals(array1: Any, array2: Any) -> bool:
    return bool(array1.equals(array2))


def _warn_ignored(ctx: Engine, kinds: list[Any], obj: Any) -> None:
    r"""Warn that the settings are ignored without NumPy, once per
    setting for a comparison, if a column has a numeric or duration
    type.

    A nested structure can contain many Arrow objects, and the same
    warning for each of them would flood the output.
    """
    if is_numpy_available() or not any(_is_quantity(kind) for kind in kinds):
        return
    for name in ("equal_nan", "atol", "rtol"):
        if getattr(ctx.config, name):
            ctx.warn_once(name, f"{name} is ignored because it is not supported for {type(obj)}")


def get_default_testers() -> dict[type, BaseEqualityTester[Any]]:
    r"""Get the default equality testers of the pyarrow objects.

    Returns:
        The equality testers of the pyarrow objects and their associated types.

    Notes:
        This function is called by the default registry the first time it
        needs the equality testers of the pyarrow objects.
    """
    if not is_pyarrow_available():
        return {}

    array = TableEqualityTester(pyarrow_array)
    table = TableEqualityTester(pyarrow_table)
    pyarrow = cast("Any", pa)
    return {
        pyarrow.Array: array,
        pyarrow.ChunkedArray: array,
        pyarrow.RecordBatch: table,
        pyarrow.Table: table,
    }

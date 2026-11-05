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
        r"""Indicate if two tables have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        _warn_ignored(ctx, self._table.schema.types, self._table)
        return bool(
            self._table.schema.equals(other._table.schema)
            and self._table.num_rows == other._table.num_rows
        )

    def structure_difference(self, other: PyarrowTableTable, ctx: Engine) -> str:  # noqa: ARG002
        r"""Say what differs in the structure of two tables.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            A short text, for example ``"numbers of rows differ: 3 vs 2"``.
        """
        t1, t2 = self._table, other._table
        if t1.num_rows != t2.num_rows:
            return f"numbers of rows differ: {t1.num_rows} vs {t2.num_rows}"
        if t1.schema.names != t2.schema.names:
            return f"columns differ: {t1.schema.names} vs {t2.schema.names}"
        pairs = [
            f"{field.name}: {field.type} vs {other_field.type}"
            for field, other_field in zip(t1.schema, t2.schema, strict=True)
            if field.type != other_field.type
        ]
        if not pairs:
            return "the nullable flags of the columns differ"
        shown = ", ".join(pairs[:3])
        more = f" (+{len(pairs) - 3} more)" if len(pairs) > 3 else ""
        return f"types differ: {shown}{more}"

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of each table column.
        """
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
        r"""Indicate if two arrays have the same structure.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        _warn_ignored(ctx, [self._array.type], self._array)
        return bool(self._array.type == other._array.type and len(self._array) == len(other._array))

    def structure_difference(self, other: PyarrowArrayTable, ctx: Engine) -> str:  # noqa: ARG002
        r"""Say what differs in the structure of two arrays.

        Args:
            other: The other table.
            ctx: The ``Engine`` of the comparison.

        Returns:
            A short text, for example ``"types differ: int64 vs double"``.
        """
        a1, a2 = self._array, other._array
        if len(a1) != len(a2):
            return f"lengths differ: {len(a1)} vs {len(a2)}"
        return f"types differ: {a1.type} vs {a2.type}"

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns.

        Yields:
            The column of the array.
        """
        yield _column(None, self._array)


def pyarrow_table(table: Any) -> PyarrowTableTable:
    r"""Adapt a ``pyarrow.Table`` or a ``pyarrow.RecordBatch`` to a
    table.

    Args:
        table: The table or record batch to adapt.

    Returns:
        The table.
    """
    return PyarrowTableTable(table)


def pyarrow_array(array: Any) -> PyarrowArrayTable:
    r"""Adapt a ``pyarrow.Array`` or a ``pyarrow.ChunkedArray`` to a
    table.

    Args:
        array: The array or chunked array to adapt.

    Returns:
        The table.
    """
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

    Args:
        name: The name of the column.
        array: The array of the column.

    Returns:
        The column.
    """
    kind = array.type
    if is_numpy_available() and _is_quantity(kind):
        return Column(
            name,
            "duration" if pa.types.is_duration(kind) else "numeric",
            _values(array, kind),
            _nulls(array),
        )
    return Column(name, "exact", array, equals=_arrow_equals, difference=_arrow_difference)


def _is_quantity(kind: Any) -> bool:
    r"""Indicate if an Arrow type is compared with the tolerance.

    Args:
        kind: The Arrow type.

    Returns:
        ``True`` for an integer, floating-point or duration type,
            otherwise ``False``.
    """
    return bool(
        pa.types.is_floating(kind) or pa.types.is_integer(kind) or pa.types.is_duration(kind)
    )


def _to_numpy(array: Any) -> Any:
    r"""Convert an array or a chunked array to a NumPy array.

    ``ChunkedArray.to_numpy`` does not accept ``zero_copy_only`` in the
    old versions of pyarrow.

    Args:
        array: The array or chunked array, without null values for
            the types that cannot hold them in NumPy.

    Returns:
        The NumPy array.
    """
    if isinstance(array, pa.ChunkedArray):
        return array.to_numpy()
    return array.to_numpy(zero_copy_only=False)


def _values(array: Any, kind: Any) -> Any:
    r"""Return the values of an array as a NumPy array, where the null
    values are replaced by zero.

    The durations are filled as integers, because the old versions of
    pyarrow have no ``coalesce`` kernel for the duration type.

    Args:
        array: The array or chunked array.
        kind: The Arrow type of the array: an integer, floating-point
            or duration type.

    Returns:
        The NumPy array.
    """
    if pa.types.is_duration(kind):
        values = _to_numpy(pc.fill_null(array.cast(pa.int64()), 0))
        return values.astype(f"timedelta64[{kind.unit}]")
    return _to_numpy(pc.fill_null(array, pa.scalar(0, type=kind)))


def _nulls(array: Any) -> Any:
    r"""Return the mask of the null values of an array.

    Args:
        array: The array.

    Returns:
        A NumPy boolean array that is ``True`` at the null values.
    """
    return _to_numpy(array.is_null())


def _arrow_equals(array1: Any, array2: Any) -> bool:
    r"""Indicate if two arrays are equal for Arrow.

    Args:
        array1: The first array.
        array2: The second array.

    Returns:
        ``True`` if the arrays are equal, otherwise ``False``.
    """
    return bool(array1.equals(array2))


def _arrow_difference(array1: Any, array2: Any) -> tuple[int, Any, Any] | None:
    r"""Find the first row where two arrays differ.

    Args:
        array1: The first array.
        array2: The second array, with the same type and length.

    Returns:
        ``(row, value, other value)`` for the first row that differs, or
            ``None`` if the row cannot be found (the nested values are
            not searched).
    """
    if pa.types.is_nested(array1.type):
        return None
    # The kernels are called by name: pyarrow generates their Python wrappers at import.
    different = pc.call_function(
        "or",
        [
            pc.fill_null(pc.call_function("not_equal", [array1, array2]), False),
            pc.call_function("xor", [array1.is_null(), array2.is_null()]),
        ],
    )
    row = pc.index(different, True).as_py()
    if row < 0:
        return None
    return row, array1[row].as_py(), array2[row].as_py()


def _warn_ignored(ctx: Engine, kinds: list[Any], obj: Any) -> None:
    r"""Warn that the settings are ignored without NumPy, once per
    setting for a comparison, if a column has a numeric or duration
    type.

    A nested structure can contain many Arrow objects, and the same
    warning for each of them would flood the output.

    Args:
        ctx: The ``Engine`` of the comparison.
        kinds: The Arrow types of the columns.
        obj: The compared object, named in the warning.
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

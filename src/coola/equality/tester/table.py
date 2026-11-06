r"""Implement the equality tester of the tables.

A table is a data frame, a series, an index or an array of pandas,
polars or pyarrow. The three libraries run the same algorithm: check the
structure, then compare the columns one by one, each by its kind, and
stop at the first column that differs. The algorithm is written once in
``TableEqualityTester``; an adapter converts an object of a library to
the ``Table`` protocol.
"""

from __future__ import annotations

__all__ = ["Column", "Table", "TableEqualityTester", "column_close", "column_difference"]

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from coola.equality.tester.base import BaseEqualityTester

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from coola.equality.engine import Engine


@dataclass(frozen=True, slots=True)
class Column:
    r"""Describe a column of a table.

    Args:
        name: The name of the column (``None`` for a series, an index
            or an array).
        kind: The kind of the column. A ``"numeric"`` column (integers,
            floats, complex numbers) and a ``"duration"`` column are
            compared with the tolerance and ``equal_nan`` of the
            comparison. An ``"exact"`` column (booleans, strings,
            categories, instants, decimals, nested values, ...) is
            compared by the library.
        values: The values of the column. A NumPy array for a numeric
            and a duration column, the column of the library for an
            exact column.
        nulls: The mask of the null values of a numeric or a duration
            column, or ``None`` if the nulls are not masked. The masks
            must be equal, then the values are compared (a null is
            replaced by zero in ``values``).
        equals: The function that compares the ``values`` of two exact
            columns (``==`` by default).
        difference: The function that finds the first row where the
            ``values`` of two exact columns differ, and returns
            ``(row, value, other value)`` or ``None`` if it cannot be
            found. It is only called to explain a difference. If
            ``None``, the row is not reported for this column.
    """

    name: object
    kind: Literal["numeric", "duration", "exact"]
    values: Any
    nulls: Any = None
    equals: Callable[[Any, Any], bool] = operator.eq
    difference: Callable[[Any, Any], tuple[int, Any, Any] | None] | None = None


class Table(Protocol):
    r"""Define what an adapter returns for an object of a library.

    Attributes:
        kind_name: The plural word used in the reasons (``"tables"``,
            ``"series"``, ``"arrays"``, ...).

    An adapter can also define ``structure_difference(other, ctx)``,
    which returns a short text that says what differs in the structure
    (for example ``"dtypes differ: a: int64 vs float64"``). It is called
    only to explain a difference found by ``same_structure``.
    """

    kind_name: str

    def same_structure(self, other: Any, ctx: Engine) -> bool:
        r"""Indicate if two tables have the same structure (shape, names,
        data types, index, ...).

        The engine gives the settings of the comparison (``equal_nan``)
        and receives the warnings.

        Args:
            other: The other table, of the same kind.
            ctx: The ``Engine`` of the comparison.

        Returns:
            ``True`` if the structures are the same, otherwise
                ``False``.
        """
        ...

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns, converted lazily: a column is
        converted only when it is reached."""
        ...


class TableEqualityTester(BaseEqualityTester[Any]):
    r"""Implement an equality tester for the tables of pandas, polars and
    pyarrow.

    The structure is compared first, then the columns one by one, each
    by its kind (see ``Column``). The comparison stops at the first
    column that differs. A table is a leaf for the engine: the
    difference is reported as a whole. The reason says what differs in
    the structure (the shapes, the data types, the index, the names of
    the columns, ...) or, for the values, the name of the first column
    that differs, the first row that differs and the two values. The
    row is found only when the difference is explained.

    Args:
        adapter: The function that converts an object to a ``Table``.

    Example:
        ```pycon
        >>> import pandas as pd
        >>> from coola.equality import compare
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import TableEqualityTester, get_default_registry
        >>> from coola.equality.tester.pandas import pandas_frame
        >>> engine = Engine(EqualityConfig())
        >>> tester = TableEqualityTester(pandas_frame)
        >>> tester.objects_are_equal(
        ...     pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2]}), engine
        ... )
        True
        >>> tester.objects_are_equal(
        ...     pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 3]}), engine
        ... )
        False
        >>> print(compare(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 3]})).reason)
        tables have different values: column 'a' differs at row 1: 2 vs 3

        ```
    """

    def __init__(self, adapter: Callable[[Any], Table]) -> None:
        self._adapter = adapter

    def _display_fields(self) -> dict[str, Any]:
        return {"adapter": getattr(self._adapter, "__name__", repr(self._adapter))}

    def objects_are_equal(self, actual: Any, expected: Any, ctx: Engine, /) -> bool:
        table, other = self._adapter(actual), self._adapter(expected)
        if not table.same_structure(other, ctx):
            return ctx.fail(lambda: _structure_reason(table, other, ctx))
        for column, other_column in zip(table.columns(), other.columns(), strict=True):
            if not column_close(column, other_column, ctx):
                found = column_difference(column, other_column, ctx) if ctx.explaining else None
                name = "" if column.name is None else f": column {column.name!r} differs"
                if found is None:
                    return ctx.fail(f"{table.kind_name} have different values{name}")
                row, value, other_value = found
                reason = (
                    f"{table.kind_name} have different values{name} at row {row}: "
                    f"{value!r} vs {other_value!r}"
                )
                return ctx.fail(reason)
        return True


def _structure_reason(table: Table, other: Table, ctx: Engine) -> str:
    r"""Return the reason of a difference of structure.

    Args:
        table: The first table.
        other: The second table.
        ctx: The engine of the comparison.

    Returns:
        The reason, with what differs if the adapter says it.
    """
    reason = f"{table.kind_name} have a different structure"
    describe = getattr(table, "structure_difference", None)
    if describe is None:
        return reason
    return f"{reason}: {describe(other, ctx)}"


def column_close(column: Column, other: Column, ctx: Engine) -> bool:
    r"""Indicate if two columns with the same name and data type are equal
    within the tolerance of the comparison.

    Args:
        column: The first column.
        other: The second column.
        ctx: The engine of the comparison.

    Returns:
        ``True`` if the two columns are equal, otherwise ``False``.

    Example:
        ```pycon
        >>> from coola.equality import Tolerance
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> from coola.equality.tester.table import Column, column_close
        >>> engine = Engine(EqualityConfig(tolerance=Tolerance(atol=0.1)))
        >>> column_close(
        ...     Column("a", "numeric", np.array([1.0, 2.0])),
        ...     Column("a", "numeric", np.array([1.0, 2.05])),
        ...     engine,
        ... )
        True

        ```
    """
    if column.kind == "exact":
        return column.equals(column.values, other.values)
    from coola.equality._helpers.numpy import array_equal  # noqa: PLC0415

    if column.nulls is not None and not bool((column.nulls == other.nulls).all()):
        return False
    return array_equal(column.values, other.values, ctx.config)


def column_difference(column: Column, other: Column, ctx: Engine) -> tuple[int, Any, Any] | None:
    r"""Find the first row where two columns differ.

    The columns must have the same name, data type and length. A null
    value is shown as ``None``.

    Args:
        column: The first column.
        other: The second column.
        ctx: The engine of the comparison.

    Returns:
        ``(row, value, other value)`` for the first row that differs,
            or ``None`` if the row cannot be found.

    Example:
        ```pycon
        >>> import numpy as np
        >>> from coola.equality.config import EqualityConfig
        >>> from coola.equality.engine import Engine
        >>> from coola.equality.tester import get_default_registry
        >>> from coola.equality.tester.table import Column, column_difference
        >>> engine = Engine(EqualityConfig())
        >>> column_difference(
        ...     Column("a", "numeric", np.array([1.0, 2.0])),
        ...     Column("a", "numeric", np.array([1.0, 3.0])),
        ...     engine,
        ... )
        (1, 2.0, 3.0)

        ```
    """
    if column.kind == "exact":
        if column.difference is None:
            return None
        found = column.difference(column.values, other.values)
        if found is None:
            return None
        row, value, other_value = found
        return row, _plain(value), _plain(other_value)
    from coola.equality._helpers.numpy import array_close  # noqa: PLC0415

    different = ~array_close(column.values, other.values, ctx.config)
    if column.nulls is not None:
        different = different | (column.nulls != other.nulls)
    rows = different.nonzero()[0]
    if len(rows) == 0:
        return None
    row = int(rows[0])
    return row, _item(column, row), _item(other, row)


def _plain(value: Any) -> Any:
    r"""Convert a NumPy scalar to a Python scalar, so that its ``repr``
    is short.

    Args:
        value: The value.

    Returns:
        The Python scalar if ``value`` is a NumPy scalar, otherwise
            ``value``.
    """
    if type(value).__module__ == "numpy":
        return value.item()
    return value


def _item(column: Column, row: int) -> Any:
    r"""Return a value of a numeric or duration column as a Python
    object.

    Args:
        column: The column.
        row: The row.

    Returns:
        ``None`` for a null value, otherwise the value.
    """
    if column.nulls is not None and column.nulls[row]:
        return None
    return column.values[row].item()

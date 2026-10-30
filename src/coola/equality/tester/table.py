r"""Implement the equality tester of the tables.

A table is a data frame, a series, an index or an array of pandas,
polars or pyarrow. The three libraries run the same algorithm: check the
structure, then compare the columns one by one, each by its kind, and
stop at the first column that differs. The algorithm is written once in
``TableEqualityTester``; an adapter converts an object of a library to
the ``Table`` protocol.
"""

from __future__ import annotations

__all__ = ["Column", "Table", "TableEqualityTester", "column_close"]

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
    """

    name: object
    kind: Literal["numeric", "duration", "exact"]
    values: Any
    nulls: Any = None
    equals: Callable[[Any, Any], bool] = operator.eq


class Table(Protocol):
    r"""Define what an adapter returns for an object of a library.

    Attributes:
        kind_name: The plural word used in the reasons (``"tables"``,
            ``"series"``, ``"arrays"``, ...).
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

    def columns(self) -> Iterator[Column]:
        r"""Iterate over the columns, converted lazily: a column is
        converted only when it is reached."""


class TableEqualityTester(BaseEqualityTester[Any]):
    r"""Implement an equality tester for the tables of pandas, polars and
    pyarrow.

    The structure is compared first, then the columns one by one, each
    by its kind (see ``Column``). The comparison stops at the first
    column that differs. A table is a leaf for the engine: the
    difference is reported as a whole, with the name of the first column
    that differs, and no row. The user finds the row with the tools of
    the library (``DataFrame.compare``, ``polars.testing``,
    ``Table.equals``).

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
        tables have different values: column 'a' differs

        ```
    """

    def __init__(self, adapter: Callable[[Any], Table]) -> None:
        self._adapter = adapter

    def _display_fields(self) -> dict[str, Any]:
        return {"adapter": getattr(self._adapter, "__name__", repr(self._adapter))}

    def objects_are_equal(self, actual: Any, expected: Any, ctx: Engine, /) -> bool:
        table, other = self._adapter(actual), self._adapter(expected)
        if not table.same_structure(other, ctx):
            return ctx.fail(f"{table.kind_name} have a different structure")
        for column, other_column in zip(table.columns(), other.columns(), strict=True):
            if not column_close(column, other_column, ctx):
                name = "" if column.name is None else f": column {column.name!r} differs"
                return ctx.fail(f"{table.kind_name} have different values{name}")
        return True


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

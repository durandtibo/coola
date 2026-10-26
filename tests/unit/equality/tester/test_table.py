from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality import compare, objects_are_allclose
from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester import EqualityTesterRegistry, TableEqualityTester, get_default_registry
from coola.equality.tester.table import Column, column_close
from coola.equality.tolerance import Tolerance
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING:
    from collections.abc import Iterator

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@dataclass
class Grid:
    r"""A table made of columns, used to test the tester without a
    library."""

    columns: list[Column]
    structure: str = "same"
    seen: int = 0


class GridTable:
    kind_name = "grids"

    def __init__(self, grid: Grid) -> None:
        self._grid = grid

    def same_structure(self, other: GridTable, ctx: Engine) -> bool:  # noqa: ARG002
        return self._grid.structure == other._grid.structure

    def columns(self) -> Iterator[Column]:
        for column in self._grid.columns:
            self._grid.seen += 1
            yield column


def grid_table(grid: Grid) -> GridTable:
    return GridTable(grid)


def numeric(name: object, values: list[float]) -> Column:
    return Column(name, "numeric", np.array(values))


@pytest.fixture
def engine() -> Engine:
    return Engine(EqualityConfig())


def grid_registry() -> EqualityTesterRegistry:
    registry = get_default_registry().copy()
    registry.register(Grid, TableEqualityTester(grid_table), exist_ok=True)
    return registry


def run(actual: Grid, expected: Grid, config: EqualityConfig | None = None) -> bool:
    return Engine(replace(config or EqualityConfig(), registry=grid_registry())).run(
        actual, expected
    )


def reason(actual: Grid, expected: Grid) -> str | None:
    return compare(actual, expected, registry=grid_registry()).reason


#########################################
#     Tests for TableEqualityTester     #
#########################################


def test_table_equality_tester_equal_true() -> None:
    assert TableEqualityTester(grid_table) == TableEqualityTester(grid_table)


def test_table_equality_tester_equal_false_different_adapter() -> None:
    assert TableEqualityTester(grid_table) != TableEqualityTester(GridTable)


def test_table_equality_tester_equal_false_different_type() -> None:
    assert TableEqualityTester(grid_table) != 42


def test_table_equality_tester_repr() -> None:
    assert repr(TableEqualityTester(grid_table)).startswith("TableEqualityTester(")


def test_table_equality_tester_str() -> None:
    assert str(TableEqualityTester(grid_table)).startswith("TableEqualityTester(")


@numpy_available
def test_table_equality_tester_objects_are_equal_true() -> None:
    assert run(Grid([numeric("a", [1, 2])]), Grid([numeric("a", [1, 2])]))


@numpy_available
def test_table_equality_tester_objects_are_equal_true_no_column() -> None:
    assert run(Grid([]), Grid([]))


@numpy_available
def test_table_equality_tester_objects_are_equal_false_different_values() -> None:
    assert not run(Grid([numeric("a", [1, 2])]), Grid([numeric("a", [1, 3])]))


@numpy_available
def test_table_equality_tester_objects_are_equal_false_different_structure() -> None:
    assert not run(Grid([numeric("a", [1])]), Grid([numeric("a", [1])], structure="other"))


@numpy_available
def test_table_equality_tester_objects_are_equal_true_tolerance() -> None:
    assert run(
        Grid([numeric("a", [1.0, 2.0])]),
        Grid([numeric("a", [1.0, 2.05])]),
        EqualityConfig(tolerance=Tolerance(atol=0.1)),
    )


@numpy_available
def test_table_equality_tester_objects_are_equal_true_equal_nan() -> None:
    assert run(
        Grid([numeric("a", [float("nan")])]),
        Grid([numeric("a", [float("nan")])]),
        EqualityConfig(equal_nan=True),
    )


@numpy_available
def test_table_equality_tester_objects_are_equal_false_nan() -> None:
    assert not run(Grid([numeric("a", [float("nan")])]), Grid([numeric("a", [float("nan")])]))


@numpy_available
def test_table_equality_tester_stops_at_the_first_column_that_differs() -> None:
    actual = Grid([numeric("a", [1]), numeric("b", [1]), numeric("c", [1])])
    run(actual, Grid([numeric("a", [1]), numeric("b", [2]), numeric("c", [1])]))
    assert actual.seen == 2


@numpy_available
def test_table_equality_tester_reason_different_structure() -> None:
    assert (
        reason(Grid([numeric("a", [1])]), Grid([numeric("a", [1])], structure="other"))
        == "grids have a different structure"
    )


@numpy_available
def test_table_equality_tester_reason_names_the_first_column_that_differs() -> None:
    assert (
        reason(
            Grid([numeric("a", [1]), numeric("b", [1]), numeric("c", [1])]),
            Grid([numeric("a", [1]), numeric("b", [2]), numeric("c", [2])]),
        )
        == "grids have different values: column 'b' differs"
    )


@numpy_available
def test_table_equality_tester_reason_without_column_name() -> None:
    assert (
        reason(Grid([numeric(None, [1])]), Grid([numeric(None, [2])]))
        == "grids have different values"
    )


@numpy_available
def test_table_equality_tester_the_difference_is_located_at_the_table() -> None:
    result = compare(
        {"k": Grid([numeric("a", [1])])}, {"k": Grid([numeric("a", [2])])}, registry=grid_registry()
    )
    assert result.path == ("k",)


##################################
#     Tests for column_close     #
##################################


@numpy_available
def test_column_close_numeric_true(engine: Engine) -> None:
    assert column_close(numeric("a", [1, 2]), numeric("a", [1, 2]), engine)


@numpy_available
def test_column_close_numeric_false(engine: Engine) -> None:
    assert not column_close(numeric("a", [1, 2]), numeric("a", [1, 3]), engine)


@numpy_available
def test_column_close_numeric_tolerance() -> None:
    engine = Engine(EqualityConfig(tolerance=Tolerance(atol=0.5)))
    assert column_close(numeric("a", [1.0]), numeric("a", [1.4]), engine)


@numpy_available
def test_column_close_same_nulls_true(engine: Engine) -> None:
    nulls = np.array([False, True])
    assert column_close(
        Column("a", "numeric", np.array([1, 0]), nulls),
        Column("a", "numeric", np.array([1, 0]), nulls),
        engine,
    )


@numpy_available
def test_column_close_different_nulls_false(engine: Engine) -> None:
    assert not column_close(
        Column("a", "numeric", np.array([1, 0]), np.array([False, True])),
        Column("a", "numeric", np.array([1, 0]), np.array([False, False])),
        engine,
    )


@numpy_available
def test_column_close_duration_atol_is_in_seconds() -> None:
    engine = Engine(EqualityConfig(tolerance=Tolerance(atol=2.0)))
    assert column_close(
        Column("a", "duration", np.array([1], dtype="timedelta64[s]")),
        Column("a", "duration", np.array([2], dtype="timedelta64[s]")),
        engine,
    )


def test_column_close_exact_uses_the_equals_of_the_column(engine: Engine) -> None:
    assert column_close(
        Column("a", "exact", "abc", equals=lambda a, b: a.lower() == b.lower()),
        Column("a", "exact", "ABC"),
        engine,
    )


def test_column_close_exact_uses_equality_by_default(engine: Engine) -> None:
    assert not column_close(Column("a", "exact", "abc"), Column("a", "exact", "ABC"), engine)


def test_column_close_exact_ignores_the_tolerance() -> None:
    engine = Engine(EqualityConfig(tolerance=Tolerance(atol=10.0)))
    assert not column_close(Column("a", "exact", 1.0), Column("a", "exact", 2.0), engine)


@numpy_available
def test_objects_are_allclose_with_a_registered_table() -> None:
    assert objects_are_allclose(
        Grid([numeric("a", [1.0])]),
        Grid([numeric("a", [1.0000001])]),
        registry=grid_registry(),
    )

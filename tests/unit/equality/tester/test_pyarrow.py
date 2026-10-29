from __future__ import annotations

import warnings
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest

from coola.equality import compare, objects_are_allclose, objects_are_equal
from coola.equality.tester import TableEqualityTester
from coola.equality.tester.pyarrow import (
    PyarrowArrayTable,
    PyarrowTableTable,
    get_default_testers,
)
from coola.testing.marks import numpy_available, pyarrow_available
from coola.utils.imports import is_pyarrow_available
from tests.unit.equality.utils import ExamplePair

if is_pyarrow_available():
    import pyarrow as pa
else:
    pa = Mock()

NAN = float("nan")

PYARROW_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        ),
        id="array float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1, 2, 3], type=pa.int64()),
            expected=pa.array([1, 2, 3], type=pa.int64()),
        ),
        id="array int dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([True, False, True], type=pa.bool_()),
            expected=pa.array([True, False, True], type=pa.bool_()),
        ),
        id="array bool dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(["polar", "bear", "meow"], type=pa.string()),
            expected=pa.array(["polar", "bear", "meow"], type=pa.string()),
        ),
        id="array string dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
        ),
        id="array date64 dtype",
    ),
]
PYARROW_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1, 2, 3], type=pa.int64()),
            expected_message="objects have different types:",
        ),
        id="array different data types",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 4.0], type=pa.float64()),
            expected_message="arrays have",
        ),
        id="array different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected="meow",
            expected_message="objects have different types:",
        ),
        id="array different types",
    ),
]
PYARROW_ARRAY_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            atol=1.0,
        ),
        id="array atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            atol=0.1,
        ),
        id="array atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            atol=0.01,
        ),
        id="array atol=0.01",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(["polar", "bear", "meow"], type=pa.string()),
            expected=pa.array(["polar", "bear", "meow"], type=pa.string()),
            atol=0.01,
        ),
        id="array string atol",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            atol=0.01,
        ),
        id="array date64 atol",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            rtol=1.0,
        ),
        id="array rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            rtol=0.1,
        ),
        id="array rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            rtol=0.01,
        ),
        id="array rtol=0.01",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(["polar", "bear", "meow"], type=pa.string()),
            expected=pa.array(["polar", "bear", "meow"], type=pa.string()),
            rtol=0.01,
        ),
        id="array string rtol",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            rtol=0.01,
        ),
        id="array date64 rtol",
    ),
]
PYARROW_ARRAY_NOT_EQUAL_TOLERANCE = [
    # The arrays are different without tolerance and equal with it.
    # atol
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.5, 1.5, 1.5], type=pa.float64()),
            atol=1.0,
        ),
        id="array atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.05, 1.05, 1.05], type=pa.float64()),
            atol=0.1,
        ),
        id="array atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.005, 1.005, 1.005], type=pa.float64()),
            atol=0.01,
        ),
        id="array atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.5, 1.5, 1.5], type=pa.float64()),
            rtol=1.0,
        ),
        id="array rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.05, 1.05, 1.05], type=pa.float64()),
            rtol=0.1,
        ),
        id="array rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.005, 1.005, 1.005], type=pa.float64()),
            rtol=0.01,
        ),
        id="array rtol=0.01",
    ),
]

PYARROW_TABLE_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pa.table({"nums": [1, 2, 3, 4, 5]}, schema=pa.schema({"nums": pa.int64()})),
            expected=pa.table({"nums": [1, 2, 3, 4, 5]}, schema=pa.schema({"nums": pa.int64()})),
        ),
        id="table 1 column",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
        ),
        id="table 2 columns",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {
                    "nums": [1, 2, 3, 4, 5],
                    "float": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "chars": ["a", "b", "c", "d", "e"],
                },
                schema=pa.schema({"nums": pa.int64(), "float": pa.float64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {
                    "nums": [1, 2, 3, 4, 5],
                    "float": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "chars": ["a", "b", "c", "d", "e"],
                },
                schema=pa.schema({"nums": pa.int64(), "float": pa.float64(), "chars": pa.string()}),
            ),
        ),
        id="table 3 columns",
    ),
]
PYARROW_TABLE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {"nums": [1.0, 2.0, 3.0, 4.0, 5.0], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.float64(), "chars": pa.string()}),
            ),
            expected_message="tables have",
        ),
        id="table different dtypes",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {"nums": [1, 2, 3, 4, 6], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected_message="tables have",
        ),
        id="table different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {"nums2": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums2": pa.int64(), "chars": pa.string()}),
            ),
            expected_message="tables have",
        ),
        id="table different columns",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected_message="objects have different types:",
        ),
        id="table different types",
    ),
]

PYARROW_EQUAL = PYARROW_ARRAY_EQUAL + PYARROW_TABLE_EQUAL
PYARROW_NOT_EQUAL = PYARROW_ARRAY_NOT_EQUAL + PYARROW_TABLE_NOT_EQUAL
PYARROW_EQUAL_TOLERANCE = PYARROW_ARRAY_EQUAL_TOLERANCE + PYARROW_ARRAY_NOT_EQUAL_TOLERANCE
PYARROW_NOT_EQUAL_TOLERANCE = [
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.5, 1.5, 1.5], type=pa.float64()),
            atol=0.1,
        ),
        id="array atol too small",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.5, 1.5, 1.5], type=pa.float64()),
            rtol=0.1,
        ),
        id="array rtol too small",
    ),
]

ARRAY_EQUAL = [
    pytest.param(pa.array([1, 2, 3]), pa.array([1, 2, 3]), id="integers"),
    pytest.param(pa.array([1.0, None]), pa.array([1.0, None]), id="float with null"),
    pytest.param(pa.array([1, None]), pa.array([1, None]), id="integer with null"),
    pytest.param(pa.array(["a", None]), pa.array(["a", None]), id="strings"),
    pytest.param(pa.array([True, None]), pa.array([True, None]), id="bool"),
    pytest.param(pa.array([[1.0], None]), pa.array([[1.0], None]), id="list"),
    pytest.param(pa.array([], type=pa.float64()), pa.array([], type=pa.float64()), id="empty"),
    pytest.param(pa.chunked_array([[1, 2], [3]]), pa.chunked_array([[1], [2, 3]]), id="chunks"),
    pytest.param(
        pa.array([1, None], type=pa.duration("s")),
        pa.array([1, None], type=pa.duration("s")),
        id="duration",
    ),
]

ARRAY_NOT_EQUAL = [
    pytest.param(pa.array([1, 2]), pa.array([1, 3]), id="values"),
    pytest.param(pa.array([1, 2]), pa.array([1.0, 2.0]), id="type"),
    pytest.param(pa.array([1, 2]), pa.array([1]), id="length"),
    pytest.param(pa.array([1, None]), pa.array([None, 1]), id="null position"),
    pytest.param(pa.array(["a"]), pa.array(["b"]), id="strings"),
    pytest.param(pa.array([True]), pa.array([False]), id="bool"),
    pytest.param(pa.array([[1.0]]), pa.array([[2.0]]), id="list"),
    pytest.param(
        pa.array([1], type=pa.duration("s")), pa.array([2], type=pa.duration("s")), id="duration"
    ),
]

TABLE_EQUAL = [
    pytest.param(pa.table({"a": [1, 2]}), pa.table({"a": [1, 2]}), id="1 column"),
    pytest.param(
        pa.table({"a": [1, 2], "b": ["x", None]}),
        pa.table({"a": [1, 2], "b": ["x", None]}),
        id="2 columns",
    ),
    pytest.param(pa.record_batch({"a": [1, 2]}), pa.record_batch({"a": [1, 2]}), id="record batch"),
    pytest.param(
        pa.table({"a": [1]}).replace_schema_metadata({"k": "v"}),
        pa.table({"a": [1]}),
        id="metadata is ignored",
    ),
]

TABLE_NOT_EQUAL = [
    pytest.param(pa.table({"a": [1, 2]}), pa.table({"a": [1, 3]}), id="values"),
    pytest.param(pa.table({"a": [1, 2]}), pa.table({"b": [1, 2]}), id="column names"),
    pytest.param(pa.table({"a": [1, 2]}), pa.table({"a": [1.0, 2.0]}), id="types"),
    pytest.param(pa.table({"a": [1, 2]}), pa.table({"a": [1]}), id="rows"),
    pytest.param(pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]}), id="record batch"),
    pytest.param(pa.table({"a": [1]}), pa.record_batch({"a": [1]}), id="table and batch"),
]


########################################
#     Tests for the pyarrow arrays     #
########################################


@pyarrow_available
@pytest.mark.parametrize(("actual", "expected"), ARRAY_EQUAL)
def test_array_equal(actual: pa.Array, expected: pa.Array) -> None:
    assert objects_are_equal(actual, expected)


@pyarrow_available
@pytest.mark.parametrize(("actual", "expected"), ARRAY_NOT_EQUAL)
def test_array_not_equal(actual: pa.Array, expected: pa.Array) -> None:
    assert not objects_are_equal(actual, expected)


@pyarrow_available
@pytest.mark.parametrize(("actual", "expected"), ARRAY_NOT_EQUAL)
def test_array_not_equal_symmetric(actual: pa.Array, expected: pa.Array) -> None:
    assert not objects_are_equal(expected, actual)


@pyarrow_available
def test_array_not_equal_to_a_chunked_array() -> None:
    assert not objects_are_equal(pa.array([1]), pa.chunked_array([[1]]))


@pyarrow_available
def test_array_nan_is_not_equal_by_default() -> None:
    assert not objects_are_equal(pa.array([NAN]), pa.array([NAN]))


@pyarrow_available
def test_array_nan_is_equal_with_equal_nan() -> None:
    assert objects_are_equal(pa.array([NAN]), pa.array([NAN]), equal_nan=True)


@pyarrow_available
def test_array_null_is_not_nan() -> None:
    assert not objects_are_equal(pa.array([None], type=pa.float64()), pa.array([NAN]))


@pyarrow_available
def test_array_nested_nan_is_not_equal() -> None:
    assert not objects_are_equal(pa.array([[NAN]]), pa.array([[NAN]]), equal_nan=True)


@pyarrow_available
def test_array_tolerance_true() -> None:
    assert objects_are_allclose(pa.array([1.0, 2.0]), pa.array([1.0, 2.05]), atol=0.1, rtol=0.0)


@pyarrow_available
def test_array_tolerance_false() -> None:
    assert not objects_are_allclose(pa.array([1.0]), pa.array([1.5]), atol=0.1, rtol=0.0)


@pyarrow_available
def test_array_tolerance_with_nulls() -> None:
    assert objects_are_allclose(pa.array([1.0, None]), pa.array([1.05, None]), atol=0.1)


@pyarrow_available
def test_array_tolerance_on_durations_is_in_seconds() -> None:
    assert objects_are_allclose(
        pa.array([1], type=pa.duration("s")), pa.array([2], type=pa.duration("s")), atol=1.5
    )


@pyarrow_available
def test_array_tolerance_does_not_apply_to_nested_values() -> None:
    assert not objects_are_allclose(pa.array([[1.0]]), pa.array([[1.0001]]), atol=0.1)


@pyarrow_available
def test_array_tolerance_does_not_apply_to_booleans() -> None:
    assert not objects_are_allclose(pa.array([True]), pa.array([False]), atol=1.0)


@pyarrow_available
def test_array_integers_above_2_53_are_exact() -> None:
    assert not objects_are_allclose(pa.array([2**62]), pa.array([2**62 + 1]), atol=0.5, rtol=0.0)


@pyarrow_available
def test_array_compare_different_values() -> None:
    assert compare(pa.array([1]), pa.array([2])).reason == "arrays have different values"


@pyarrow_available
def test_array_compare_different_structure() -> None:
    assert compare(pa.array([1]), pa.array([1, 2])).reason == "arrays have a different structure"


########################################
#     Tests for the pyarrow tables     #
########################################


@pyarrow_available
@pytest.mark.parametrize(("actual", "expected"), TABLE_EQUAL)
def test_table_equal(actual: pa.Table, expected: pa.Table) -> None:
    assert objects_are_equal(actual, expected)


@pyarrow_available
@pytest.mark.parametrize(("actual", "expected"), TABLE_NOT_EQUAL)
def test_table_not_equal(actual: pa.Table, expected: pa.Table) -> None:
    assert not objects_are_equal(actual, expected)


@pyarrow_available
@pytest.mark.parametrize(("actual", "expected"), TABLE_NOT_EQUAL)
def test_table_not_equal_symmetric(actual: pa.Table, expected: pa.Table) -> None:
    assert not objects_are_equal(expected, actual)


@pyarrow_available
def test_table_tolerance() -> None:
    assert objects_are_allclose(pa.table({"a": [1.0]}), pa.table({"a": [1.05]}), atol=0.1)


@pyarrow_available
def test_table_nan_is_equal_with_equal_nan() -> None:
    assert objects_are_equal(pa.table({"a": [NAN]}), pa.table({"a": [NAN]}), equal_nan=True)


@pyarrow_available
def test_table_chunked_columns() -> None:
    table = pa.Table.from_batches([pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]})])
    assert objects_are_equal(table, pa.table({"a": [1, 2]}))


@pyarrow_available
def test_table_compare_different_values() -> None:
    result = compare(pa.table({"a": [1], "b": [1]}), pa.table({"a": [1], "b": [2]}))
    assert result.reason == "tables have different values: column 'b' differs"


@pyarrow_available
def test_table_compare_different_structure() -> None:
    result = compare(pa.table({"a": [1]}), pa.table({"b": [1]}))
    assert result.reason == "tables have a different structure"


@pyarrow_available
def test_table_compare_in_a_dict() -> None:
    result = compare({"k": pa.table({"a": [1]})}, {"k": pa.table({"a": [2]})})
    assert result.path == ("k",)


@pyarrow_available
def test_table_compare_the_whole_result() -> None:
    result = compare(pa.table({"a": [1]}), pa.table({"a": [2]}))
    assert (result.equal, result.path, result.reason) == (
        False,
        (),
        "tables have different values: column 'a' differs",
    )


############################################
#     Tests for pyarrow without NumPy     #
#########################################


@pyarrow_available
@numpy_available
@pytest.mark.parametrize("name", ["equal_nan", "atol", "rtol"])
def test_a_setting_is_ignored_with_a_warning_without_numpy(name: str) -> None:
    settings = {"equal_nan": {"equal_nan": True}, "atol": {"atol": 0.1}, "rtol": {"rtol": 0.1}}
    kwargs: dict[str, Any] = {"rtol": 0.0, "atol": 0.0, **settings[name]}
    with (
        patch("coola.equality.tester.pyarrow.is_numpy_available", return_value=False),
        pytest.warns(RuntimeWarning, match=f"{name} is ignored"),
    ):
        compare(pa.array([1.0]), pa.array([1.0]), **kwargs)


@pyarrow_available
@numpy_available
def test_the_warning_is_emitted_once_per_comparison_without_numpy() -> None:
    with (
        patch("coola.equality.tester.pyarrow.is_numpy_available", return_value=False),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        objects_are_allclose(
            pa.table({"a": [1.0], "b": [2.0]}), pa.table({"a": [1.0], "b": [2.0]}), atol=0.1
        )
    assert len([w for w in caught if "atol is ignored" in str(w.message)]) == 1


@pyarrow_available
@numpy_available
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_columns_are_compared_by_arrow_without_numpy() -> None:
    with patch("coola.equality.tester.pyarrow.is_numpy_available", return_value=False):
        assert not objects_are_allclose(pa.array([1.0]), pa.array([1.05]), atol=0.1)


###################################
#     Tests for the adapters     #
###################################


@pyarrow_available
@pytest.mark.parametrize(
    ("table", "name"),
    [
        (PyarrowTableTable(pa.table({"a": [1]})), "tables"),
        (PyarrowArrayTable(pa.array([1])), "arrays"),
    ],
)
def test_kind_name(table: Any, name: str) -> None:
    assert table.kind_name == name


@pyarrow_available
@pytest.mark.parametrize("dtype", [pa.Array, pa.ChunkedArray, pa.RecordBatch, pa.Table])
def test_get_default_testers(dtype: type) -> None:
    assert isinstance(get_default_testers()[dtype], TableEqualityTester)


def test_get_default_testers_without_pyarrow() -> None:
    with patch("coola.equality.tester.pyarrow.is_pyarrow_available", return_value=False):
        assert get_default_testers() == {}

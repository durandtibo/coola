from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import Mock, patch

import pytest

from coola.equality import compare, objects_are_allclose, objects_are_equal
from coola.equality.tester import TableEqualityTester
from coola.equality.tester.polars import (
    PolarsFrameTable,
    PolarsLazyFrameTable,
    PolarsSeriesTable,
    get_default_testers,
)
from coola.testing.marks import polars_available
from coola.utils.imports import is_polars_available
from tests.unit.equality.utils import ExamplePair

if is_polars_available():
    import polars as pl
else:
    pl = Mock()

NAN = float("nan")

POLARS_DATAFRAME_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({}),
            expected=pl.DataFrame({}),
        ),
        id="0 column",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col": [1, 2, 3]}),
            expected=pl.DataFrame({"col": [1, 2, 3]}),
        ),
        id="1 column",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            expected=pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
        ),
        id="2 columns",
    ),
]

POLARS_DATAFRAME_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col": [1, 2, 3]}),
            expected=pl.DataFrame({"col": [1, 2, 4]}),
            expected_message="tables have",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col1": [1, 2, 3]}),
            expected=pl.DataFrame({"col2": [1, 2, 3]}),
            expected_message="tables have",
        ),
        id="different column names",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col1": [1, 2, 3]}),
            expected=pl.Series([1, 2, 3]),
            expected_message="objects have different types:",
        ),
        id="different column names",
    ),
]

POLARS_DATAFRAME_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.DataFrame({"col": [1.5, 1.5, 0.5]}),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.DataFrame({"col": [1.0, 1.05, 0.95]}),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.DataFrame({"col": [1.0, 1.005, 0.995]}),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.DataFrame({"col": [1.0, 1.5, 0.5]}),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.DataFrame({"col": [1.0, 1.05, 0.95]}),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.DataFrame({"col": [1.0, 1.005, 0.995]}),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

POLARS_LAZYFRAME_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({}),
            expected=pl.LazyFrame({}),
        ),
        id="0 column",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col": [1, 2, 3]}),
            expected=pl.LazyFrame({"col": [1, 2, 3]}),
        ),
        id="1 column",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            expected=pl.LazyFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
        ),
        id="2 columns",
    ),
]

POLARS_LAZYFRAME_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col": [1, 2, 3]}),
            expected=pl.LazyFrame({"col": [1, 2, 4]}),
            expected_message="tables have",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col1": [1, 2, 3]}),
            expected=pl.LazyFrame({"col2": [1, 2, 3]}),
            expected_message="tables have",
        ),
        id="different column names",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col1": [1, 2, 3]}),
            expected=pl.Series([1, 2, 3]),
            expected_message="objects have different types:",
        ),
        id="different column names",
    ),
]

POLARS_LAZYFRAME_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.LazyFrame({"col": [1.5, 1.5, 0.5]}),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.LazyFrame({"col": [1.0, 1.05, 0.95]}),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.LazyFrame({"col": [1.0, 1.005, 0.995]}),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.LazyFrame({"col": [1.0, 1.5, 0.5]}),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.LazyFrame({"col": [1.0, 1.05, 0.95]}),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pl.LazyFrame({"col": [1.0, 1.005, 0.995]}),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

POLARS_SERIES_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pl.Series([]),
            expected=pl.Series([]),
        ),
        id="empty",
    ),
    pytest.param(ExamplePair(actual=pl.Series([1, 2, 3]), expected=pl.Series([1, 2, 3])), id="int"),
    pytest.param(
        ExamplePair(actual=pl.Series(["a", "b", "c"]), expected=pl.Series(["a", "b", "c"])),
        id="str",
    ),
]

POLARS_SERIES_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pl.Series([1, 2, 3]),
            expected=pl.Series([1, 2, 4]),
            expected_message="series have",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1, 2, 3]),
            expected=pl.Series([1, 2, 3, 4]),
            expected_message="series have",
        ),
        id="different shape",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1, 2, 3]),
            expected=pl.Series([1.0, 2.0, 3.0]),
            expected_message="series have",
        ),
        id="different data type",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1, 2, 3]),
            expected=42,
            expected_message="objects have different types:",
        ),
        id="different type",
    ),
]

POLARS_SERIES_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=pl.Series([1.0, 1.0, 1.0]),
            expected=pl.Series([1.0, 1.5, 0.5]),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1.0, 1.0, 1.0]),
            expected=pl.Series([1.0, 1.05, 0.95]),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1.0, 1.0, 1.0]),
            expected=pl.Series([1.0, 1.005, 0.995]),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pl.Series([1.0, 1.0, 1.0]),
            expected=pl.Series([1.0, 1.5, 0.5]),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1.0, 1.0, 1.0]),
            expected=pl.Series([1.0, 1.05, 0.95]),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1.0, 1.0, 1.0]),
            expected=pl.Series([1.0, 1.005, 0.995]),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

POLARS_EQUAL = POLARS_SERIES_EQUAL + POLARS_DATAFRAME_EQUAL + POLARS_LAZYFRAME_EQUAL
POLARS_NOT_EQUAL = POLARS_SERIES_NOT_EQUAL + POLARS_DATAFRAME_NOT_EQUAL + POLARS_LAZYFRAME_NOT_EQUAL
POLARS_EQUAL_TOLERANCE = (
    POLARS_SERIES_EQUAL_TOLERANCE
    + POLARS_DATAFRAME_EQUAL_TOLERANCE
    + POLARS_LAZYFRAME_EQUAL_TOLERANCE
)

FRAME_EQUAL = [
    pytest.param(pl.DataFrame({}), pl.DataFrame({}), id="0 column"),
    pytest.param(pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [1, 2]}), id="1 column"),
    pytest.param(
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        id="2 columns",
    ),
    pytest.param(pl.DataFrame({"a": [1, None]}), pl.DataFrame({"a": [1, None]}), id="null"),
    pytest.param(pl.DataFrame({"a": [True, None]}), pl.DataFrame({"a": [True, None]}), id="bool"),
    pytest.param(
        pl.DataFrame({"a": [[1.0, 2.0]]}), pl.DataFrame({"a": [[1.0, 2.0]]}), id="list column"
    ),
    pytest.param(
        pl.DataFrame({"a": [{"x": 1}]}), pl.DataFrame({"a": [{"x": 1}]}), id="struct column"
    ),
    pytest.param(
        pl.DataFrame({"a": [timedelta(seconds=1), None]}),
        pl.DataFrame({"a": [timedelta(seconds=1), None]}),
        id="duration with null",
    ),
]

FRAME_NOT_EQUAL = [
    pytest.param(pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [1, 3]}), id="values"),
    pytest.param(pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"b": [1, 2]}), id="column names"),
    pytest.param(pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [1.0, 2.0]}), id="dtypes"),
    pytest.param(pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [1]}), id="heights"),
    pytest.param(
        pl.DataFrame({"a": [1], "b": [2]}), pl.DataFrame({"b": [2], "a": [1]}), id="column order"
    ),
    pytest.param(
        pl.DataFrame({"a": [1, None]}), pl.DataFrame({"a": [None, 1]}), id="null position"
    ),
    pytest.param(pl.DataFrame({"a": ["x"]}), pl.DataFrame({"a": ["y"]}), id="strings"),
    pytest.param(pl.DataFrame({"a": [True]}), pl.DataFrame({"a": [False]}), id="bool"),
    pytest.param(pl.DataFrame({"a": [[1.0]]}), pl.DataFrame({"a": [[2.0]]}), id="list column"),
    pytest.param(
        pl.DataFrame({"a": [timedelta(seconds=1)]}),
        pl.DataFrame({"a": [timedelta(seconds=2)]}),
        id="duration",
    ),
]

SERIES_EQUAL = [
    pytest.param(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), id="integers"),
    pytest.param(pl.Series("s", [1.0, None]), pl.Series("s", [1.0, None]), id="name and null"),
    pytest.param(pl.Series([], dtype=pl.Float64), pl.Series([], dtype=pl.Float64), id="empty"),
    pytest.param(pl.Series(["a", None]), pl.Series(["a", None]), id="strings"),
]

SERIES_NOT_EQUAL = [
    pytest.param(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), id="values"),
    pytest.param(pl.Series("a", [1]), pl.Series("b", [1]), id="name"),
    pytest.param(pl.Series([1]), pl.Series([1.0]), id="dtype"),
    pytest.param(pl.Series([1]), pl.Series([1, 2]), id="length"),
]


###########################################
#     Tests for the polars DataFrames     #
###########################################


@polars_available
@pytest.mark.parametrize(("actual", "expected"), FRAME_EQUAL)
def test_frame_equal(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    assert objects_are_equal(actual, expected)


@polars_available
@pytest.mark.parametrize(("actual", "expected"), FRAME_NOT_EQUAL)
def test_frame_not_equal(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    assert not objects_are_equal(actual, expected)


@polars_available
@pytest.mark.parametrize(("actual", "expected"), FRAME_NOT_EQUAL)
def test_frame_not_equal_symmetric(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    assert not objects_are_equal(expected, actual)


@polars_available
def test_frame_not_equal_to_a_series() -> None:
    assert not objects_are_equal(pl.DataFrame({"a": [1]}), pl.Series("a", [1]))


@polars_available
def test_frame_nan_is_not_equal_by_default() -> None:
    assert not objects_are_equal(pl.DataFrame({"a": [NAN]}), pl.DataFrame({"a": [NAN]}))


@polars_available
def test_frame_nan_is_equal_with_equal_nan() -> None:
    assert objects_are_equal(pl.DataFrame({"a": [NAN]}), pl.DataFrame({"a": [NAN]}), equal_nan=True)


@polars_available
def test_frame_null_is_not_nan() -> None:
    assert not objects_are_equal(
        pl.DataFrame({"a": [None]}, schema={"a": pl.Float64}), pl.DataFrame({"a": [NAN]})
    )


@polars_available
def test_frame_tolerance_true() -> None:
    assert objects_are_allclose(
        pl.DataFrame({"a": [1.0, 2.0]}), pl.DataFrame({"a": [1.0, 2.05]}), atol=0.1, rtol=0.0
    )


@polars_available
def test_frame_tolerance_false() -> None:
    assert not objects_are_allclose(
        pl.DataFrame({"a": [1.0]}), pl.DataFrame({"a": [1.5]}), atol=0.1, rtol=0.0
    )


@polars_available
def test_frame_tolerance_with_nulls() -> None:
    assert objects_are_allclose(
        pl.DataFrame({"a": [1.0, None]}), pl.DataFrame({"a": [1.05, None]}), atol=0.1
    )


@polars_available
def test_frame_tolerance_on_durations_is_in_seconds() -> None:
    assert objects_are_allclose(
        pl.DataFrame({"a": [timedelta(seconds=1)]}),
        pl.DataFrame({"a": [timedelta(seconds=2)]}),
        atol=1.5,
    )


@polars_available
def test_frame_tolerance_does_not_apply_to_nested_values() -> None:
    assert not objects_are_allclose(
        pl.DataFrame({"a": [[1.0]]}), pl.DataFrame({"a": [[1.0001]]}), atol=0.1
    )


@polars_available
def test_frame_tolerance_does_not_apply_to_booleans() -> None:
    assert not objects_are_allclose(
        pl.DataFrame({"a": [True]}), pl.DataFrame({"a": [False]}), atol=1.0
    )


@polars_available
def test_frame_integers_above_2_53_are_exact() -> None:
    assert not objects_are_allclose(
        pl.DataFrame({"a": [2**62]}), pl.DataFrame({"a": [2**62 + 1]}), atol=0.5, rtol=0.0
    )


@polars_available
def test_frame_compare_different_values() -> None:
    result = compare(pl.DataFrame({"a": [1], "b": [1]}), pl.DataFrame({"a": [1], "b": [2]}))
    assert result.reason == "tables have different values: column 'b' differs"


@polars_available
def test_frame_compare_different_structure() -> None:
    result = compare(pl.DataFrame({"a": [1]}), pl.DataFrame({"b": [1]}))
    assert result.reason == "tables have a different structure"


@polars_available
def test_frame_compare_in_a_dict() -> None:
    result = compare({"k": pl.DataFrame({"a": [1]})}, {"k": pl.DataFrame({"a": [2]})})
    assert result.path == ("k",)


@polars_available
def test_frame_compare_the_whole_result() -> None:
    result = compare(pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [2]}))
    assert (result.equal, result.path, result.reason) == (
        False,
        (),
        "tables have different values: column 'a' differs",
    )


###########################################
#     Tests for the polars LazyFrames     #
###########################################


@polars_available
def test_lazy_frame_equal() -> None:
    assert objects_are_equal(pl.LazyFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 2]}))


@polars_available
def test_lazy_frame_not_equal_values() -> None:
    assert not objects_are_equal(pl.LazyFrame({"a": [1, 2]}), pl.LazyFrame({"a": [1, 3]}))


@polars_available
def test_lazy_frame_not_equal_schema() -> None:
    assert not objects_are_equal(pl.LazyFrame({"a": [1]}), pl.LazyFrame({"a": [1.0]}))


@polars_available
def test_lazy_frame_not_equal_height() -> None:
    assert not objects_are_equal(pl.LazyFrame({"a": [1]}), pl.LazyFrame({"a": [1, 2]}))


@polars_available
def test_lazy_frame_ignores_the_query_plan() -> None:
    assert objects_are_equal(
        pl.LazyFrame({"a": [1, 2]}).select(pl.col("a")), pl.LazyFrame({"a": [1, 2]})
    )


@polars_available
def test_lazy_frame_tolerance() -> None:
    assert objects_are_allclose(pl.LazyFrame({"a": [1.0]}), pl.LazyFrame({"a": [1.05]}), atol=0.1)


@polars_available
def test_lazy_frame_not_equal_to_a_data_frame() -> None:
    assert not objects_are_equal(pl.LazyFrame({"a": [1]}), pl.DataFrame({"a": [1]}))


@polars_available
def test_lazy_frame_compare_different_values() -> None:
    result = compare(pl.LazyFrame({"a": [1]}), pl.LazyFrame({"a": [2]}))
    assert result.reason == "tables have different values: column 'a' differs"


@polars_available
def test_lazy_frame_compare_different_schema() -> None:
    result = compare(pl.LazyFrame({"a": [1]}), pl.LazyFrame({"a": [1.0]}))
    assert result.reason == "tables have a different structure"


@polars_available
def test_lazy_frame_collects_each_frame_once() -> None:
    table = PolarsLazyFrameTable(pl.LazyFrame({"a": [1]}))
    assert table.collect() is table.collect()


########################################
#     Tests for the polars Series     #
########################################


@polars_available
@pytest.mark.parametrize(("actual", "expected"), SERIES_EQUAL)
def test_series_equal(actual: pl.Series, expected: pl.Series) -> None:
    assert objects_are_equal(actual, expected)


@polars_available
@pytest.mark.parametrize(("actual", "expected"), SERIES_NOT_EQUAL)
def test_series_not_equal(actual: pl.Series, expected: pl.Series) -> None:
    assert not objects_are_equal(actual, expected)


@polars_available
@pytest.mark.parametrize(("actual", "expected"), SERIES_NOT_EQUAL)
def test_series_not_equal_symmetric(actual: pl.Series, expected: pl.Series) -> None:
    assert not objects_are_equal(expected, actual)


@polars_available
def test_series_nan_is_not_equal_by_default() -> None:
    assert not objects_are_equal(pl.Series([NAN]), pl.Series([NAN]))


@polars_available
def test_series_nan_is_equal_with_equal_nan() -> None:
    assert objects_are_equal(pl.Series([NAN]), pl.Series([NAN]), equal_nan=True)


@polars_available
def test_series_tolerance() -> None:
    assert objects_are_allclose(pl.Series([1.0, 2.0]), pl.Series([1.0, 2.05]), atol=0.1)


@polars_available
def test_series_compare_different_values() -> None:
    assert compare(pl.Series([1]), pl.Series([2])).reason == "series have different values"


@polars_available
def test_series_compare_different_structure() -> None:
    assert compare(pl.Series([1]), pl.Series([1, 2])).reason == "series have a different structure"


###################################
#     Tests for the adapters     #
###################################


@polars_available
@pytest.mark.parametrize(
    ("table", "name"),
    [
        (PolarsFrameTable(pl.DataFrame()), "tables"),
        (PolarsLazyFrameTable(pl.LazyFrame()), "tables"),
        (PolarsSeriesTable(pl.Series([1])), "series"),
    ],
)
def test_kind_name(table: Any, name: str) -> None:
    assert table.kind_name == name


@polars_available
@pytest.mark.parametrize("dtype", [pl.DataFrame, pl.LazyFrame, pl.Series])
def test_get_default_testers(dtype: type) -> None:
    assert isinstance(get_default_testers()[dtype], TableEqualityTester)


def test_get_default_testers_without_polars() -> None:
    with patch("coola.equality.tester.polars.is_polars_available", return_value=False):
        assert get_default_testers() == {}

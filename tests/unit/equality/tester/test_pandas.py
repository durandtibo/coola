from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from coola.equality import compare, objects_are_allclose, objects_are_equal
from coola.equality.config import EqualityConfig
from coola.equality.engine import Engine
from coola.equality.tester import TableEqualityTester, get_default_registry
from coola.equality.tester.pandas import (
    PandasArrayTable,
    PandasFrameTable,
    PandasIndexTable,
    PandasSeriesTable,
    _dtype_label,
    _dtypes_difference,
    _index_difference,
    _index_difference_row,
    _same_dtype,
    _same_dtype_list,
    _series_difference,
    _short_list,
    get_default_testers,
)
from coola.testing.marks import pandas_available
from coola.utils.imports import is_pandas_available
from tests.unit.equality.utils import ExamplePair

if is_pandas_available():
    import pandas as pd
else:
    pd = Mock()

NAN = float("nan")

CAT_AB = pd.CategoricalDtype(["a", "b"])
CAT_BA = pd.CategoricalDtype(["b", "a"])

PANDAS_DATAFRAME_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({}),
            expected=pd.DataFrame({}),
        ),
        id="0 column",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col": [1, 2, 3]}),
            expected=pd.DataFrame({"col": [1, 2, 3]}),
        ),
        id="1 column",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            expected=pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
        ),
        id="2 columns",
    ),
]

PANDAS_DATAFRAME_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col": [1, 2, 3]}),
            expected=pd.DataFrame({"col": [1, 2, 4]}),
            expected_message="tables have",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col1": [1, 2, 3]}),
            expected=pd.DataFrame({"col2": [1, 2, 3]}),
            expected_message="tables have",
        ),
        id="different column names",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col1": [1, 2, 3]}),
            expected=pd.Series([1, 2, 3]),
            expected_message="objects have different types:",
        ),
        id="different column names",
    ),
]

PANDAS_DATAFRAME_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pd.DataFrame({"col": [1.5, 1.5, 0.5]}),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pd.DataFrame({"col": [1.0, 1.05, 0.95]}),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pd.DataFrame({"col": [1.0, 1.005, 0.995]}),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pd.DataFrame({"col": [1.0, 1.5, 0.5]}),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pd.DataFrame({"col": [1.0, 1.05, 0.95]}),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col": [1.0, 1.0, 1.0]}),
            expected=pd.DataFrame({"col": [1.0, 1.005, 0.995]}),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

PANDAS_SERIES_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pd.Series(data=[], dtype=object),
            expected=pd.Series(data=[], dtype=object),
        ),
        id="empty",
    ),
    pytest.param(ExamplePair(actual=pd.Series([1, 2, 3]), expected=pd.Series([1, 2, 3])), id="int"),
    pytest.param(
        ExamplePair(actual=pd.Series(["a", "b", "c"]), expected=pd.Series(["a", "b", "c"])),
        id="str",
    ),
]

PANDAS_SERIES_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pd.Series([1, 2, 3]),
            expected=pd.Series([1, 2, 4]),
            expected_message="series have",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1, 2, 3]),
            expected=pd.Series([1, 2, 3, 4]),
            expected_message="series have",
        ),
        id="different shape",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1, 2, 3]),
            expected=pd.Series([1.0, 2.0, 3.0]),
            expected_message="series have",
        ),
        id="different data type",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1, 2, 3]),
            expected=42,
            expected_message="objects have different types:",
        ),
        id="different type",
    ),
]

PANDAS_SERIES_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=pd.Series([1.0, 1.0, 1.0]),
            expected=pd.Series([1.0, 1.5, 0.5]),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1.0, 1.0, 1.0]),
            expected=pd.Series([1.0, 1.05, 0.95]),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1.0, 1.0, 1.0]),
            expected=pd.Series([1.0, 1.005, 0.995]),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pd.Series([1.0, 1.0, 1.0]),
            expected=pd.Series([1.0, 1.5, 0.5]),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1.0, 1.0, 1.0]),
            expected=pd.Series([1.0, 1.05, 0.95]),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1.0, 1.0, 1.0]),
            expected=pd.Series([1.0, 1.005, 0.995]),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

PANDAS_EQUAL = PANDAS_SERIES_EQUAL + PANDAS_DATAFRAME_EQUAL
PANDAS_NOT_EQUAL = PANDAS_SERIES_NOT_EQUAL + PANDAS_DATAFRAME_NOT_EQUAL
PANDAS_EQUAL_TOLERANCE = PANDAS_SERIES_EQUAL_TOLERANCE + PANDAS_DATAFRAME_EQUAL_TOLERANCE

FRAME_EQUAL = [
    pytest.param(pd.DataFrame({}), pd.DataFrame({}), id="0 column"),
    pytest.param(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2]}), id="1 column"),
    pytest.param(
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        id="2 columns",
    ),
    pytest.param(
        pd.DataFrame({"a": [1, 2]}, index=["x", "y"]),
        pd.DataFrame({"a": [1, 2]}, index=["x", "y"]),
        id="index",
    ),
    pytest.param(pd.DataFrame({"a": [True, False]}), pd.DataFrame({"a": [True, False]}), id="bool"),
    pytest.param(
        pd.DataFrame({"a": [None, 1]}, dtype="Int64"),
        pd.DataFrame({"a": [None, 1]}, dtype="Int64"),
        id="nullable integers with a null",
    ),
    pytest.param(
        pd.DataFrame({"a": [None, "x"]}), pd.DataFrame({"a": [None, "x"]}), id="object with None"
    ),
    pytest.param(
        pd.DataFrame({"a": pd.to_timedelta([1, None], unit="s")}),
        pd.DataFrame({"a": pd.to_timedelta([1, None], unit="s")}),
        id="timedelta with NaT",
    ),
]

FRAME_NOT_EQUAL = [
    pytest.param(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 3]}), id="values"),
    pytest.param(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [1, 2]}), id="column names"),
    pytest.param(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1.0, 2.0]}), id="dtypes"),
    pytest.param(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1]}), id="shapes"),
    pytest.param(
        pd.DataFrame({"a": [1, 2]}, index=["x", "y"]),
        pd.DataFrame({"a": [1, 2]}, index=["x", "z"]),
        id="index",
    ),
    pytest.param(pd.DataFrame({"a": [True]}), pd.DataFrame({"a": [False]}), id="bool"),
    pytest.param(
        pd.DataFrame({"a": [None, 1]}), pd.DataFrame({"a": [1, None]}), id="null position"
    ),
    pytest.param(pd.DataFrame({"a": ["x"]}), pd.DataFrame({"a": ["y"]}), id="strings"),
    pytest.param(pd.DataFrame({"a": [[1.0]]}), pd.DataFrame({"a": [[2.0]]}), id="nested"),
    pytest.param(
        pd.DataFrame({"a": pd.to_timedelta([1], unit="s")}),
        pd.DataFrame({"a": pd.to_timedelta([2], unit="s")}),
        id="timedelta",
    ),
]

SERIES_EQUAL = [
    pytest.param(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), id="integers"),
    pytest.param(pd.Series([1.0, 2.0], name="s"), pd.Series([1.0, 2.0], name="s"), id="name"),
    pytest.param(pd.Series([], dtype=float), pd.Series([], dtype=float), id="empty"),
    pytest.param(pd.Series(["a", None]), pd.Series(["a", None]), id="strings with None"),
]

SERIES_NOT_EQUAL = [
    pytest.param(pd.Series([1, 2, 3]), pd.Series([1, 2, 4]), id="values"),
    pytest.param(pd.Series([1, 2]), pd.Series([1, 2], name="s"), id="name"),
    pytest.param(pd.Series([1, 2]), pd.Series([1.0, 2.0]), id="dtype"),
    pytest.param(pd.Series([1, 2]), pd.Series([1, 2, 3]), id="length"),
    pytest.param(pd.Series([1, 2]), pd.Series([1, 2], index=["a", "b"]), id="index"),
]


###########################################
#     Tests for the pandas DataFrames     #
###########################################


@pandas_available
@pytest.mark.parametrize(("actual", "expected"), FRAME_EQUAL)
def test_frame_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert objects_are_equal(actual, expected)


@pandas_available
@pytest.mark.parametrize(("actual", "expected"), FRAME_EQUAL)
def test_frame_equal_symmetric(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert objects_are_equal(expected, actual)


@pandas_available
@pytest.mark.parametrize(("actual", "expected"), FRAME_NOT_EQUAL)
def test_frame_not_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert not objects_are_equal(actual, expected)


@pandas_available
@pytest.mark.parametrize(("actual", "expected"), FRAME_NOT_EQUAL)
def test_frame_not_equal_symmetric(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert not objects_are_equal(expected, actual)


@pandas_available
def test_frame_not_equal_to_a_series() -> None:
    assert not objects_are_equal(pd.DataFrame({"a": [1]}), pd.Series([1]))


@pandas_available
def test_frame_nan_is_not_equal_by_default() -> None:
    assert not objects_are_equal(pd.DataFrame({"a": [NAN]}), pd.DataFrame({"a": [NAN]}))


@pandas_available
def test_frame_nan_is_equal_with_equal_nan() -> None:
    assert objects_are_equal(pd.DataFrame({"a": [NAN]}), pd.DataFrame({"a": [NAN]}), equal_nan=True)


@pandas_available
def test_frame_nan_is_equal_to_itself() -> None:
    frame = pd.DataFrame({"a": [NAN]})
    assert objects_are_equal(frame, frame)


@pandas_available
def test_frame_nullable_float_missing_value_follows_equal_nan() -> None:
    frame = pd.DataFrame({"a": pd.array([1.0, None], dtype="Float64")})
    assert not objects_are_equal(frame, frame.copy())


@pandas_available
def test_frame_nullable_float_missing_value_with_equal_nan() -> None:
    frame = pd.DataFrame({"a": pd.array([1.0, None], dtype="Float64")})
    assert objects_are_equal(frame, frame.copy(), equal_nan=True)


@pandas_available
def test_frame_index_with_nan_is_not_equal_by_default() -> None:
    frame = pd.DataFrame({"a": [1, 2]}, index=[NAN, 1.0])
    assert not objects_are_equal(frame, frame.copy())


@pandas_available
def test_frame_index_with_nan_is_equal_with_equal_nan() -> None:
    frame = pd.DataFrame({"a": [1, 2]}, index=[NAN, 1.0])
    assert objects_are_equal(frame, frame.copy(), equal_nan=True)


@pandas_available
def test_frame_tolerance_true() -> None:
    assert objects_are_allclose(
        pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"a": [1.0, 2.05]}), atol=0.1, rtol=0.0
    )


@pandas_available
def test_frame_tolerance_false() -> None:
    assert not objects_are_allclose(
        pd.DataFrame({"a": [1.0, 2.0]}), pd.DataFrame({"a": [1.0, 2.5]}), atol=0.1, rtol=0.0
    )


@pandas_available
def test_frame_tolerance_does_not_apply_to_booleans() -> None:
    assert not objects_are_allclose(
        pd.DataFrame({"a": [True]}), pd.DataFrame({"a": [False]}), atol=1.0
    )


@pandas_available
def test_frame_tolerance_does_not_apply_to_the_index() -> None:
    assert not objects_are_allclose(
        pd.DataFrame({"a": [1]}, index=[1]), pd.DataFrame({"a": [1]}, index=[2]), atol=5.0
    )


@pandas_available
def test_frame_tolerance_does_not_apply_to_nested_values() -> None:
    assert not objects_are_allclose(
        pd.DataFrame({"a": [[1.0]]}), pd.DataFrame({"a": [[1.0001]]}), atol=0.1
    )


@pandas_available
def test_frame_tolerance_on_durations_is_in_seconds() -> None:
    assert objects_are_allclose(
        pd.DataFrame({"a": pd.to_timedelta([1], unit="s")}),
        pd.DataFrame({"a": pd.to_timedelta([2], unit="s")}),
        atol=1.5,
    )


@pandas_available
def test_frame_integers_above_2_53_are_exact() -> None:
    assert not objects_are_allclose(
        pd.DataFrame({"a": [2**62]}), pd.DataFrame({"a": [2**62 + 1]}), atol=0.5, rtol=0.0
    )


@pandas_available
def test_frame_duplicate_column_names() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["a", "a"])
    assert objects_are_equal(frame, frame.copy())


@pandas_available
def test_frame_multi_index() -> None:
    index = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
    assert objects_are_equal(
        pd.DataFrame({"x": [1, 2]}, index=index), pd.DataFrame({"x": [1, 2]}, index=index.copy())
    )


@pandas_available
def test_frame_different_multi_index() -> None:
    assert not objects_are_equal(
        pd.DataFrame({"x": [1]}, index=pd.MultiIndex.from_tuples([("a", 1)])),
        pd.DataFrame({"x": [1]}, index=pd.MultiIndex.from_tuples([("a", 2)])),
    )


@pandas_available
def test_frame_does_not_modify_the_inputs() -> None:
    frame = pd.DataFrame({"a": [1.0, NAN]})
    objects_are_equal(frame, frame.copy(), equal_nan=True)
    assert frame.equals(pd.DataFrame({"a": [1.0, NAN]}))


@pandas_available
def test_frame_compare_different_values() -> None:
    result = compare(pd.DataFrame({"a": [1], "b": [1]}), pd.DataFrame({"a": [1], "b": [2]}))
    assert result.reason == "tables have different values: column 'b' differs at row 0: 1 vs 2"


@pandas_available
def test_frame_compare_different_values_is_a_leaf() -> None:
    result = compare(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}))
    assert result.path == ()


@pandas_available
def test_frame_compare_different_structure() -> None:
    result = compare(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    assert result.reason == "tables have a different structure: columns differ: ['a'] vs ['b']"


@pandas_available
def test_frame_compare_in_a_dict() -> None:
    result = compare({"prices": pd.DataFrame({"a": [1]})}, {"prices": pd.DataFrame({"a": [2]})})
    assert result.path == ("prices",)


@pandas_available
def test_frame_compare_the_whole_result() -> None:
    result = compare(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]}))
    assert (result.equal, result.path, result.reason) == (
        False,
        (),
        "tables have different values: column 'a' differs at row 0: 1 vs 2",
    )


@pandas_available
def test_frame_stops_at_the_first_column_that_differs() -> None:
    result = compare(
        pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
        pd.DataFrame({"a": [2], "b": [2], "c": [2]}),
    )
    assert result.reason == "tables have different values: column 'a' differs at row 0: 1 vs 2"


#######################################
#     Tests for the pandas Series     #
#######################################


@pandas_available
@pytest.mark.parametrize(("actual", "expected"), SERIES_EQUAL)
def test_series_equal(actual: pd.Series, expected: pd.Series) -> None:
    assert objects_are_equal(actual, expected)


@pandas_available
@pytest.mark.parametrize(("actual", "expected"), SERIES_NOT_EQUAL)
def test_series_not_equal(actual: pd.Series, expected: pd.Series) -> None:
    assert not objects_are_equal(actual, expected)


@pandas_available
@pytest.mark.parametrize(("actual", "expected"), SERIES_NOT_EQUAL)
def test_series_not_equal_symmetric(actual: pd.Series, expected: pd.Series) -> None:
    assert not objects_are_equal(expected, actual)


@pandas_available
def test_series_nan_is_not_equal_by_default() -> None:
    assert not objects_are_equal(pd.Series([NAN]), pd.Series([NAN]))


@pandas_available
def test_series_nan_is_equal_with_equal_nan() -> None:
    assert objects_are_equal(pd.Series([NAN]), pd.Series([NAN]), equal_nan=True)


@pandas_available
def test_series_name_nan_is_equal() -> None:
    assert objects_are_equal(pd.Series([1], name=NAN), pd.Series([1], name=NAN))


@pandas_available
def test_series_name_tuple_is_equal() -> None:
    assert objects_are_equal(pd.Series([1], name=("a", 1)), pd.Series([1], name=("a", 1)))


@pandas_available
def test_series_tolerance() -> None:
    assert objects_are_allclose(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.05]), atol=0.1)


@pandas_available
def test_series_tolerance_rtol() -> None:
    assert objects_are_allclose(pd.Series([100.0]), pd.Series([101.0]), rtol=0.02, atol=0.0)


@pandas_available
def test_series_compare_different_values() -> None:
    assert (
        compare(pd.Series([1]), pd.Series([2])).reason
        == "series have different values at row 0: 1 vs 2"
    )


@pandas_available
def test_series_compare_different_structure() -> None:
    assert compare(pd.Series([1]), pd.Series([1, 2])).reason == (
        "series have a different structure: shapes differ: (1,) vs (2,)"
    )


######################################
#     Tests for the pandas Index     #
######################################


@pandas_available
@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 2, 3]),
        pd.Index(["a", "b"]),
        pd.Index([1, 2], name="n"),
        pd.RangeIndex(3),
        pd.DatetimeIndex(["2020-01-01"]),
        pd.MultiIndex.from_tuples([("a", 1)]),
        pd.Index([]),
    ],
)
def test_index_equal(index: pd.Index) -> None:
    assert objects_are_equal(index, index.copy())


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(pd.Index([1, 2]), pd.Index([1, 3]), id="values"),
        pytest.param(pd.Index([1, 2]), pd.Index([1.0, 2.0]), id="dtype"),
        pytest.param(pd.Index([1, 2]), pd.Index([1, 2, 3]), id="length"),
        pytest.param(pd.Index([1, 2], name="a"), pd.Index([1, 2], name="b"), id="name"),
        pytest.param(pd.Index([1, 2]), pd.RangeIndex(1, 3), id="class"),
    ],
)
def test_index_not_equal(actual: pd.Index, expected: pd.Index) -> None:
    assert not objects_are_equal(actual, expected)


@pandas_available
def test_index_with_nan_is_not_equal_by_default() -> None:
    assert not objects_are_equal(pd.Index([1.0, NAN]), pd.Index([1.0, NAN]))


@pandas_available
def test_index_with_nan_is_equal_with_equal_nan() -> None:
    assert objects_are_equal(pd.Index([1.0, NAN]), pd.Index([1.0, NAN]), equal_nan=True)


@pandas_available
def test_multi_index_with_nan_is_not_equal_by_default() -> None:
    index = pd.MultiIndex.from_tuples([("a", 1.0), ("b", NAN)])
    assert not objects_are_equal(index, index.copy())


@pandas_available
def test_multi_index_with_nan_is_equal_with_equal_nan() -> None:
    index = pd.MultiIndex.from_tuples([("a", 1.0), ("b", NAN)])
    assert objects_are_equal(index, index.copy(), equal_nan=True)


@pandas_available
def test_index_with_none_is_equal() -> None:
    assert objects_are_equal(pd.Index(["a", None]), pd.Index(["a", None]))


@pandas_available
def test_index_has_no_tolerance() -> None:
    assert not objects_are_allclose(pd.Index([1.0]), pd.Index([1.001]), atol=1.0)


@pandas_available
def test_index_compare_different_values() -> None:
    assert (
        compare(pd.Index([1]), pd.Index([2])).reason
        == "indexes have different values at row 0: 1 vs 2"
    )


@pandas_available
def test_index_compare_different_structure() -> None:
    assert compare(pd.Index([1]), pd.Index([1, 2])).reason == (
        "indexes have a different structure: index shapes differ: (1,) vs (2,)"
    )


#######################################
#     Tests for the pandas arrays     #
#######################################


@pandas_available
@pytest.mark.parametrize(
    "array",
    [
        pd.Categorical(["a", "b"]),
        pd.array([1, None], dtype="Int64"),
        pd.array(["a", None], dtype="string"),
        pd.array([True, None], dtype="boolean"),
        pd.array([1.0, 2.0], dtype="Float64"),
        pd.period_range("2020-01", periods=2, freq="M").array,
        pd.arrays.IntervalArray.from_breaks([0, 1, 2]),
    ],
)
def test_array_equal(array: pd.api.extensions.ExtensionArray) -> None:
    assert objects_are_equal(array, array.copy())


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(pd.Categorical(["a", "b"]), pd.Categorical(["a", "c"]), id="categorical"),
        pytest.param(
            pd.array([1, 2], dtype="Int64"), pd.array([1, 3], dtype="Int64"), id="integers"
        ),
        pytest.param(pd.array([1, 2], dtype="Int64"), pd.array([1, 2], dtype="Int32"), id="dtype"),
        pytest.param(pd.array([1, 2], dtype="Int64"), pd.array([1], dtype="Int64"), id="shape"),
        pytest.param(
            pd.array([1, None], dtype="Int64"), pd.array([None, 1], dtype="Int64"), id="null"
        ),
    ],
)
def test_array_not_equal(
    actual: pd.api.extensions.ExtensionArray, expected: pd.api.extensions.ExtensionArray
) -> None:
    assert not objects_are_equal(actual, expected)


@pandas_available
def test_array_tolerance_on_nullable_integers() -> None:
    assert objects_are_allclose(
        pd.array([1, None], dtype="Int64"), pd.array([2, None], dtype="Int64"), atol=1.0
    )


@pandas_available
def test_array_nullable_floats_follow_equal_nan() -> None:
    array = pd.array([1.0, None], dtype="Float64")
    assert not objects_are_equal(array, array.copy())


@pandas_available
def test_array_compare_different_values() -> None:
    result = compare(pd.array([1], dtype="Int64"), pd.array([2], dtype="Int64"))
    assert result.reason == "arrays have different values at row 0: 1 vs 2"


@pandas_available
def test_array_compare_different_structure() -> None:
    result = compare(pd.array([1], dtype="Int64"), pd.array([1, 2], dtype="Int64"))
    assert result.reason == "arrays have a different structure: shapes differ: (1,) vs (2,)"


##################################
#     Tests for the adapters     #
##################################


@pandas_available
@pytest.mark.parametrize(
    ("table", "name"),
    [
        (PandasFrameTable(pd.DataFrame()), "tables"),
        (PandasSeriesTable(pd.Series([1])), "series"),
        (PandasIndexTable(pd.Index([1])), "indexes"),
        (PandasArrayTable(pd.array([1], dtype="Int64")), "arrays"),
    ],
)
def test_kind_name(table: Any, name: str) -> None:
    assert table.kind_name == name


@pandas_available
def test_get_default_testers_frame() -> None:
    assert isinstance(get_default_testers()[pd.DataFrame], TableEqualityTester)


@pandas_available
def test_get_default_testers_extension_array() -> None:
    assert isinstance(get_default_testers()[pd.api.extensions.ExtensionArray], TableEqualityTester)


@pandas_available
def test_the_default_registry_finds_a_multi_index() -> None:
    assert isinstance(
        get_default_registry().find_equality_tester(pd.MultiIndex), TableEqualityTester
    )


def test_get_default_testers_without_pandas() -> None:
    with patch("coola.equality.tester.pandas.is_pandas_available", return_value=False):
        assert get_default_testers() == {}


###################################################
#     Tests for the reason of the differences     #
###################################################

STRUCTURE_REASONS = [
    pytest.param(
        pd.DataFrame({"a": [1, 2]}),
        pd.DataFrame({"a": [1]}),
        "tables have a different structure: shapes differ: (2, 1) vs (1, 1)",
        id="frame shape",
    ),
    pytest.param(
        pd.DataFrame({"a": [1], "b": [1]}),
        pd.DataFrame({"a": [1.0], "b": [1]}),
        "tables have a different structure: dtypes differ: a: int64 vs float64",
        id="frame dtype",
    ),
    pytest.param(
        pd.DataFrame({"a": [1], "b": [1]}),
        pd.DataFrame({"b": [1], "a": [1]}),
        "tables have a different structure: columns differ: ['a', 'b'] vs ['b', 'a']",
        id="frame column order",
    ),
    pytest.param(
        pd.DataFrame({"a": [1, 2]}),
        pd.DataFrame({"a": [1, 2]}, index=[5, 6]),
        "tables have a different structure: index classes differ: RangeIndex vs Index",
        id="frame index class",
    ),
    pytest.param(
        pd.DataFrame({"a": [1, 2]}, index=[3, 4]),
        pd.DataFrame({"a": [1, 2]}, index=[5, 6]),
        "tables have a different structure: index values differ: [3, 4] vs [5, 6]",
        id="frame index values",
    ),
    pytest.param(
        pd.DataFrame({"a": [1, 2]}, index=pd.Index([3, 4], name="i")),
        pd.DataFrame({"a": [1, 2]}, index=[3, 4]),
        "tables have a different structure: index names differ: ['i'] vs [None]",
        id="frame index name",
    ),
    pytest.param(
        pd.DataFrame({"a": [1, 2]}, index=[3, 4]),
        pd.DataFrame({"a": [1, 2]}, index=[3.0, 4.0]),
        "tables have a different structure: index dtypes differ: int64 vs float64",
        id="frame index dtype",
    ),
    pytest.param(
        pd.DataFrame({"a": [1.0]}, index=[NAN]),
        pd.DataFrame({"a": [1.0]}, index=[NAN]),
        "tables have a different structure: index has a NaN, which is not equal to NaN "
        "(use equal_nan=True)",
        id="frame index nan",
    ),
    pytest.param(
        pd.Series([1, 2]),
        pd.Series([1]),
        "series have a different structure: shapes differ: (2,) vs (1,)",
        id="series shape",
    ),
    pytest.param(
        pd.Series([1]),
        pd.Series([1.0]),
        "series have a different structure: dtypes differ: int64 vs float64",
        id="series dtype",
    ),
    pytest.param(
        pd.Series([1], name="a"),
        pd.Series([1], name="b"),
        "series have a different structure: names differ: 'a' vs 'b'",
        id="series name",
    ),
    pytest.param(
        pd.Series([1], index=[3]),
        pd.Series([1], index=[4]),
        "series have a different structure: index values differ: [3] vs [4]",
        id="series index",
    ),
    pytest.param(
        pd.array([1], dtype="Int64"),
        pd.array([1, 2], dtype="Int64"),
        "arrays have a different structure: shapes differ: (1,) vs (2,)",
        id="array shape",
    ),
    pytest.param(
        pd.Categorical(["a"]),
        pd.Categorical(["a"], categories=["a", "b"]),
        (
            "arrays have a different structure: dtypes differ: "
            "category(categories=['a'], ordered=False) vs "
            "category(categories=['a', 'b'], ordered=False)"
        ),
        id="array dtype",
    ),
    pytest.param(
        pd.Index([1]),
        pd.Index([1, 2]),
        "indexes have a different structure: index shapes differ: (1,) vs (2,)",
        id="index shape",
    ),
    pytest.param(
        pd.Index([1], name="a"),
        pd.Index([1], name="b"),
        "indexes have a different structure: index names differ: ['a'] vs ['b']",
        id="index name",
    ),
]


@pandas_available
@pytest.mark.parametrize(("actual", "expected", "reason"), STRUCTURE_REASONS)
def test_compare_says_what_differs_in_the_structure(
    actual: Any, expected: Any, reason: str
) -> None:
    assert compare(actual, expected).reason == reason


VALUE_REASONS = [
    pytest.param(
        pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
        pd.DataFrame({"a": [1.0, 2.0, 4.0]}),
        "tables have different values: column 'a' differs at row 2: 3.0 vs 4.0",
        id="float column",
    ),
    pytest.param(
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pd.DataFrame({"a": [1, 2], "b": ["x", "z"]}),
        "tables have different values: column 'b' differs at row 1: 'y' vs 'z'",
        id="string column",
    ),
    pytest.param(
        pd.DataFrame({"a": pd.array([1, None], dtype="Int64")}),
        pd.DataFrame({"a": pd.array([1, 2], dtype="Int64")}),
        "tables have different values: column 'a' differs at row 1: None vs 2",
        id="nullable integer column",
    ),
    pytest.param(
        pd.Series([True, False]),
        pd.Series([True, True]),
        "series have different values at row 1: False vs True",
        id="bool series",
    ),
    pytest.param(
        pd.Index(["a", "b"]),
        pd.Index(["a", "c"]),
        "indexes have different values at row 1: 'b' vs 'c'",
        id="index",
    ),
]


@pandas_available
@pytest.mark.parametrize(("actual", "expected", "reason"), VALUE_REASONS)
def test_compare_says_the_first_row_that_differs(actual: Any, expected: Any, reason: str) -> None:
    assert compare(actual, expected).reason == reason


@pandas_available
def test_objects_are_equal_does_not_look_for_the_row() -> None:
    assert not objects_are_equal(pd.Series(["a", "b"]), pd.Series(["a", "c"]))


@pandas_available
def test_structure_difference_frame_more_than_three_dtypes() -> None:
    frame = pd.DataFrame({name: [1] for name in "abcde"})
    other = frame.astype("float64")
    engine = Engine(EqualityConfig())
    assert PandasFrameTable(frame).structure_difference(PandasFrameTable(other), engine) == (
        "dtypes differ: a: int64 vs float64, b: int64 vs float64, c: int64 vs float64 (+2 more)"
    )


@pandas_available
def test_structure_difference_index_table_nan() -> None:
    engine = Engine(EqualityConfig())
    index = pd.Index([1.0, NAN])
    assert PandasIndexTable(index).structure_difference(PandasIndexTable(index), engine) == (
        "index has a NaN, which is not equal to NaN (use equal_nan=True)"
    )


@pandas_available
def test_structure_difference_array_dtype() -> None:
    engine = Engine(EqualityConfig())
    table = PandasArrayTable(pd.array([1], dtype="Int64"))
    other = PandasArrayTable(pd.array([1], dtype="Int32"))
    assert table.structure_difference(other, engine) == "dtypes differ: Int64 vs Int32"


@pandas_available
def test_structure_difference_series_dtype_is_checked_before_the_name() -> None:
    engine = Engine(EqualityConfig())
    table = PandasSeriesTable(pd.Series([1], name="a"))
    other = PandasSeriesTable(pd.Series([1.0], name="b"))
    assert table.structure_difference(other, engine) == "dtypes differ: int64 vs float64"


####################################################
#     Tests for the helpers of the differences     #
####################################################


@pandas_available
def test_dtypes_difference_lists_the_columns_that_differ() -> None:
    frame, other = pd.DataFrame({"a": [1], "b": [1]}), pd.DataFrame({"a": [1], "b": [1.0]})
    assert _dtypes_difference(frame.columns, frame.dtypes, other.dtypes) == (
        "dtypes differ: b: int64 vs float64"
    )


@pandas_available
def test_dtypes_difference_shows_three_columns() -> None:
    frame = pd.DataFrame({name: [1] for name in "abcd"})
    text = _dtypes_difference(frame.columns, frame.dtypes, frame.astype("float64").dtypes)
    assert (
        text
        == "dtypes differ: a: int64 vs float64, b: int64 vs float64, c: int64 vs float64 (+1 more)"
    )


@pandas_available
def test_index_difference_columns_wording() -> None:
    engine = Engine(EqualityConfig())
    assert _index_difference("columns", pd.Index(["a"]), pd.Index(["b"]), engine) == (
        "columns differ: ['a'] vs ['b']"
    )


@pandas_available
def test_index_difference_index_wording() -> None:
    engine = Engine(EqualityConfig())
    assert _index_difference("index", pd.Index(["a"]), pd.Index(["b"]), engine) == (
        "index values differ: ['a'] vs ['b']"
    )


@pandas_available
def test_index_difference_equal_nan_does_not_report_the_nan() -> None:
    engine = Engine(EqualityConfig(equal_nan=True))
    assert _index_difference("index", pd.Index([NAN]), pd.Index([NAN]), engine) == (
        "index values differ: [nan] vs [nan]"
    )


@pandas_available
def test_index_difference_multi_index_dtypes() -> None:
    engine = Engine(EqualityConfig())
    index = pd.MultiIndex.from_arrays([[1], ["a"]])
    other = pd.MultiIndex.from_arrays([[1.0], ["a"]])
    assert _index_difference("index", index, other, engine).startswith("index dtypes differ")


@pandas_available
def test_short_list_short_index() -> None:
    assert _short_list(pd.Index([1, 2, 3])) == "[1, 2, 3]"


@pandas_available
def test_short_list_long_index() -> None:
    assert _short_list(pd.Index(range(10))) == "[0, 1, 2, 3, 4, ...]"


@pandas_available
def test_short_list_limit() -> None:
    assert _short_list(pd.Index(range(10)), limit=2) == "[0, 1, ...]"


@pandas_available
def test_series_difference_first_row() -> None:
    assert _series_difference(pd.Series(["a", "b", "c"]), pd.Series(["a", "x", "y"])) == (
        1,
        "b",
        "x",
    )


@pandas_available
def test_series_difference_ignores_the_index() -> None:
    assert _series_difference(pd.Series(["a", "b"], index=[5, 6]), pd.Series(["a", "c"])) == (
        1,
        "b",
        "c",
    )


@pandas_available
def test_series_difference_equal_series() -> None:
    assert _series_difference(pd.Series(["a", None]), pd.Series(["a", None])) is None


@pandas_available
def test_index_difference_row_first_position() -> None:
    assert _index_difference_row(pd.Index(["a", "b"]), pd.Index(["a", "c"])) == (1, "b", "c")


@pandas_available
def test_index_difference_row_equal_indexes() -> None:
    assert _index_difference_row(pd.Index(["a"]), pd.Index(["a"])) is None


##############################################
#     Tests for the order of the categories  #
##############################################


@pandas_available
def test_objects_are_equal_categorical_series_same_categories_same_order() -> None:
    assert objects_are_equal(pd.Series(["a"], dtype=CAT_AB), pd.Series(["a"], dtype=CAT_AB))


@pandas_available
def test_objects_are_equal_categorical_series_different_order_of_categories() -> None:
    assert not objects_are_equal(pd.Series(["a"], dtype=CAT_AB), pd.Series(["a"], dtype=CAT_BA))


@pandas_available
def test_objects_are_equal_categorical_series_different_ordered_flag() -> None:
    assert not objects_are_equal(
        pd.Series(["a"], dtype=CAT_AB),
        pd.Series(["a"], dtype=pd.CategoricalDtype(["a", "b"], ordered=True)),
    )


@pandas_available
def test_objects_are_equal_categorical_frame_different_order_of_categories() -> None:
    assert not objects_are_equal(
        pd.DataFrame({"a": pd.Series(["a"], dtype=CAT_AB)}),
        pd.DataFrame({"a": pd.Series(["a"], dtype=CAT_BA)}),
    )


@pandas_available
def test_objects_are_equal_categorical_array_different_order_of_categories() -> None:
    assert not objects_are_equal(
        pd.Categorical(["a"], dtype=CAT_AB), pd.Categorical(["a"], dtype=CAT_BA)
    )


@pandas_available
def test_objects_are_equal_categorical_index_different_order_of_categories() -> None:
    assert not objects_are_equal(
        pd.CategoricalIndex(["a"], dtype=CAT_AB), pd.CategoricalIndex(["a"], dtype=CAT_BA)
    )


@pandas_available
def test_objects_are_equal_multi_index_with_categorical_level_different_order() -> None:
    assert not objects_are_equal(
        pd.MultiIndex.from_arrays([pd.Categorical(["a"], dtype=CAT_AB)]),
        pd.MultiIndex.from_arrays([pd.Categorical(["a"], dtype=CAT_BA)]),
    )


@pandas_available
def test_same_dtype_same_categories_same_order() -> None:
    assert _same_dtype(CAT_AB, pd.CategoricalDtype(["a", "b"]))


@pandas_available
def test_same_dtype_different_order_of_categories() -> None:
    assert not _same_dtype(CAT_AB, CAT_BA)


@pandas_available
def test_same_dtype_different_ordered_flag() -> None:
    assert not _same_dtype(CAT_AB, pd.CategoricalDtype(["a", "b"], ordered=True))


@pandas_available
def test_same_dtype_different_categories() -> None:
    assert not _same_dtype(CAT_AB, pd.CategoricalDtype(["a", "c"]))


@pandas_available
def test_same_dtype_categorical_and_object() -> None:
    assert not _same_dtype(CAT_AB, pd.Series(["a"]).dtype)


@pandas_available
def test_same_dtype_equal_numpy_dtypes() -> None:
    assert _same_dtype(pd.Series([1]).dtype, pd.Series([2]).dtype)


@pandas_available
def test_same_dtype_different_numpy_dtypes() -> None:
    assert not _same_dtype(pd.Series([1]).dtype, pd.Series([1.0]).dtype)


@pandas_available
def test_same_dtype_list_same_dtypes() -> None:
    assert _same_dtype_list([CAT_AB, "int64"], [CAT_AB, "int64"])


@pandas_available
def test_same_dtype_list_different_order_of_categories() -> None:
    assert not _same_dtype_list([CAT_AB], [CAT_BA])


@pandas_available
def test_same_dtype_list_different_lengths() -> None:
    assert not _same_dtype_list([CAT_AB], [CAT_AB, CAT_AB])


@pandas_available
def test_dtype_label_categorical() -> None:
    assert _dtype_label(CAT_BA) == "category(categories=['b', 'a'], ordered=False)"


@pandas_available
def test_dtype_label_ordered_categorical() -> None:
    assert (
        _dtype_label(pd.CategoricalDtype(["a"], ordered=True))
        == "category(categories=['a'], ordered=True)"
    )


@pandas_available
def test_dtype_label_other_dtype() -> None:
    assert _dtype_label(pd.Series([1]).dtype) == "int64"


@pandas_available
def test_dtypes_difference_categorical_order() -> None:
    assert _dtypes_difference(["a"], [CAT_AB], [CAT_BA]) == (
        "dtypes differ: a: category(categories=['a', 'b'], ordered=False) vs "
        "category(categories=['b', 'a'], ordered=False)"
    )

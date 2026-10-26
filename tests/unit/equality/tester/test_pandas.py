from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    PandasDataFrameEqualityTester,
    PandasExtensionArrayEqualityTester,
    PandasIndexEqualityTester,
    PandasSeriesEqualityTester,
)
from coola.testing.fixtures import pandas_available
from coola.utils.imports import MissingPackageError, is_pandas_available
from tests.unit.equality.utils import ExamplePair, messages

if is_pandas_available():
    import pandas as pd
else:
    pd = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


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
            expected_message="pandas.DataFrames are different:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col1": [1, 2, 3]}),
            expected=pd.DataFrame({"col2": [1, 2, 3]}),
            expected_message="pandas.DataFrames are different:",
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
            expected_message="pandas.Series are different:",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1, 2, 3]),
            expected=pd.Series([1, 2, 3, 4]),
            expected_message="pandas.Series are different:",
        ),
        id="different shape",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1, 2, 3]),
            expected=pd.Series([1.0, 2.0, 3.0]),
            expected_message="pandas.Series are different:",
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


###################################################
#     Tests for PandasDataFrameEqualityTester     #
###################################################


@pandas_available
def test_pandas_dataframe_equality_tester_repr() -> None:
    assert repr(PandasDataFrameEqualityTester()).startswith("PandasDataFrameEqualityTester(")


@pandas_available
def test_pandas_dataframe_equality_tester_str() -> None:
    assert str(PandasDataFrameEqualityTester()).startswith("PandasDataFrameEqualityTester(")


@pandas_available
def test_pandas_dataframe_equality_tester_equal_true() -> None:
    assert PandasDataFrameEqualityTester().equal(PandasDataFrameEqualityTester())


@pandas_available
def test_pandas_dataframe_equality_tester_equal_false_different_type() -> None:
    assert not PandasDataFrameEqualityTester().equal(123)


@pandas_available
def test_pandas_dataframe_equality_tester_equal_false_different_type_child() -> None:
    class Child(PandasDataFrameEqualityTester): ...

    assert not PandasDataFrameEqualityTester().equal(Child())


@pandas_available
def test_pandas_dataframe_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    val = pd.DataFrame({"col": [1, 2, 3]})
    assert PandasDataFrameEqualityTester().objects_are_equal(val, val, config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PandasDataFrameEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PandasDataFrameEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PandasDataFrameEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PandasDataFrameEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_NOT_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PandasDataFrameEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_NOT_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PandasDataFrameEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_NOT_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PandasDataFrameEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_NOT_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PandasDataFrameEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert messages(config)[0].startswith(example.expected_message)


@pandas_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_pandas_dataframe_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        PandasDataFrameEqualityTester().objects_are_equal(
            actual=pd.DataFrame({"col": [1, float("nan"), 3]}),
            expected=pd.DataFrame({"col": [1, float("nan"), 3]}),
            config=config,
        )
        == equal_nan
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL_TOLERANCE)
def test_pandas_dataframe_equality_tester_objects_are_equal_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PandasDataFrameEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


def test_pandas_dataframe_equality_tester_no_pandas() -> None:
    with (
        patch("coola.utils.imports.pandas.is_pandas_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'pandas' package is required but not installed."
        ),
    ):
        PandasDataFrameEqualityTester()


################################################
#     Tests for PandasSeriesEqualityTester     #
################################################


@pandas_available
def test_pandas_series_equality_tester_repr() -> None:
    assert repr(PandasSeriesEqualityTester()).startswith("PandasSeriesEqualityTester(")


@pandas_available
def test_pandas_series_equality_tester_str() -> None:
    assert str(PandasSeriesEqualityTester()).startswith("PandasSeriesEqualityTester(")


@pandas_available
def test_pandas_series_equality_tester_equal_true() -> None:
    assert PandasSeriesEqualityTester().equal(PandasSeriesEqualityTester())


@pandas_available
def test_pandas_series_equality_tester_equal_false_different_type() -> None:
    assert not PandasSeriesEqualityTester().equal(123)


@pandas_available
def test_pandas_series_equality_tester_equal_false_different_type_child() -> None:
    class Child(PandasSeriesEqualityTester): ...

    assert not PandasSeriesEqualityTester().equal(Child())


@pandas_available
def test_pandas_series_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    series = pd.Series([1, 2, 3])
    assert PandasSeriesEqualityTester().objects_are_equal(series, series, config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PandasSeriesEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PandasSeriesEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PandasSeriesEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PandasSeriesEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PandasSeriesEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PandasSeriesEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PandasSeriesEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PandasSeriesEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert messages(config)[0].startswith(example.expected_message)


@pandas_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_pandas_series_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        PandasSeriesEqualityTester().objects_are_equal(
            actual=pd.Series([0.0, float("nan"), float("nan"), 1.2]),
            expected=pd.Series([0.0, float("nan"), float("nan"), 1.2]),
            config=config,
        )
        == equal_nan
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL_TOLERANCE)
def test_pandas_series_equality_tester_objects_are_equal_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PandasSeriesEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


def test_pandas_series_equality_tester_no_pandas() -> None:
    with (
        patch("coola.utils.imports.pandas.is_pandas_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'pandas' package is required but not installed."
        ),
    ):
        PandasSeriesEqualityTester()


##############################################
#     Tests for PandasIndexEqualityTester    #
##############################################


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(pd.Index([1, 2, 3]), pd.Index([1, 2, 3]), id="int"),
        pytest.param(pd.Index(["a", "b"], name="x"), pd.Index(["a", "b"], name="x"), id="named"),
        pytest.param(
            pd.MultiIndex.from_tuples([(1, "a"), (2, "b")]),
            pd.MultiIndex.from_tuples([(1, "a"), (2, "b")]),
            id="multi",
        ),
        pytest.param(
            pd.date_range("2020-01-01", periods=3),
            pd.date_range("2020-01-01", periods=3),
            id="datetime",
        ),
    ],
)
def test_pandas_index_equality_tester_objects_are_equal_true(
    actual: pd.Index, expected: pd.Index, config: EqualityConfig
) -> None:
    assert PandasIndexEqualityTester().objects_are_equal(actual, expected, config)


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(pd.Index([1, 2, 3]), pd.Index([1, 2, 4]), id="different values"),
        pytest.param(pd.Index([1, 2, 3]), pd.Index([1, 2]), id="different length"),
        pytest.param(pd.Index([1, 2]), pd.Index([1.0, 2.0]), id="different dtype"),
        pytest.param(pd.Index([1, 2], name="a"), pd.Index([1, 2], name="b"), id="different name"),
        pytest.param(
            pd.MultiIndex.from_tuples([(1, "a")]),
            pd.MultiIndex.from_tuples([(1, "b")]),
            id="multi",
        ),
    ],
)
def test_pandas_index_equality_tester_objects_are_equal_false(
    actual: pd.Index, expected: pd.Index, config: EqualityConfig
) -> None:
    assert not PandasIndexEqualityTester().objects_are_equal(actual, expected, config)


@pandas_available
def test_pandas_index_equality_tester_objects_are_equal_false_explaining(
    config: EqualityConfig,
) -> None:
    config.reports = []
    assert not PandasIndexEqualityTester().objects_are_equal(pd.Index([1]), pd.Index([2]), config)


@pandas_available
def test_pandas_index_equality_tester_objects_are_equal_false_explaining_reports(
    config: EqualityConfig,
) -> None:
    config.reports = []
    PandasIndexEqualityTester().objects_are_equal(pd.Index([1]), pd.Index([2]), config)
    assert "pandas.Index are different:" in messages(config)[-1]


PANDAS_NAN_INDEXES = [
    pytest.param(pd.Index([1.0, float("nan")]), id="index"),
    pytest.param(pd.MultiIndex.from_tuples([(1.0, float("nan"))]), id="multi index"),
]


@pandas_available
@pytest.mark.parametrize("index", PANDAS_NAN_INDEXES)
def test_pandas_index_equality_tester_objects_are_equal_nan(index: pd.Index) -> None:
    tester = PandasIndexEqualityTester()
    assert not tester.objects_are_equal(index, index.copy(), EqualityConfig())


@pandas_available
@pytest.mark.parametrize("index", PANDAS_NAN_INDEXES)
def test_pandas_index_equality_tester_objects_are_equal_equal_nan(index: pd.Index) -> None:
    tester = PandasIndexEqualityTester()
    assert tester.objects_are_equal(index, index.copy(), EqualityConfig(equal_nan=True))


@pandas_available
def test_pandas_index_equality_tester_objects_are_equal_without_tolerance() -> None:
    tester = PandasIndexEqualityTester()
    actual, expected = pd.Index([1.0, 2.0]), pd.Index([1.0, 2.001])
    assert not tester.objects_are_equal(actual, expected, EqualityConfig())


@pandas_available
def test_pandas_index_equality_tester_objects_are_equal_no_tolerance() -> None:
    tester = PandasIndexEqualityTester()
    actual, expected = pd.Index([1.0, 2.0]), pd.Index([1.0, 2.001])
    assert not tester.objects_are_equal(actual, expected, EqualityConfig(atol=0.01))


def test_pandas_index_equality_tester_no_pandas() -> None:
    with (
        patch("coola.utils.imports.pandas.is_pandas_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'pandas' package is required but not installed."
        ),
    ):
        PandasIndexEqualityTester()


#########################################################
#     Tests for PandasExtensionArrayEqualityTester      #
#########################################################


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(pd.Categorical(["a", "b"]), pd.Categorical(["a", "b"]), id="categorical"),
        pytest.param(
            pd.array([1, 2], dtype="Int64"), pd.array([1, 2], dtype="Int64"), id="nullable int"
        ),
        pytest.param(
            pd.array(["a", "b"], dtype="string"), pd.array(["a", "b"], dtype="string"), id="string"
        ),
    ],
)
def test_pandas_extension_array_equality_tester_objects_are_equal_true(
    actual: object, expected: object, config: EqualityConfig
) -> None:
    assert PandasExtensionArrayEqualityTester().objects_are_equal(actual, expected, config)


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(pd.Categorical(["a", "b"]), pd.Categorical(["a", "c"]), id="values"),
        pytest.param(pd.Categorical(["a", "b"]), pd.Categorical(["a"]), id="length"),
        pytest.param(pd.array([1, 2], dtype="Int64"), pd.array([1, 3], dtype="Int64"), id="int"),
    ],
)
def test_pandas_extension_array_equality_tester_objects_are_equal_false(
    actual: object, expected: object, config: EqualityConfig
) -> None:
    assert not PandasExtensionArrayEqualityTester().objects_are_equal(actual, expected, config)


@pandas_available
def test_pandas_extension_array_equality_tester_objects_are_equal_false_explaining(
    config: EqualityConfig,
) -> None:
    config.reports = []
    assert not PandasExtensionArrayEqualityTester().objects_are_equal(
        pd.Categorical(["a"]), pd.Categorical(["b"]), config
    )


@pandas_available
def test_pandas_extension_array_equality_tester_objects_are_equal_false_explaining_reports(
    config: EqualityConfig,
) -> None:
    config.reports = []
    PandasExtensionArrayEqualityTester().objects_are_equal(
        pd.Categorical(["a"]), pd.Categorical(["b"]), config
    )
    assert "pandas arrays are different:" in messages(config)[-1]


@pandas_available
def test_pandas_extension_array_equality_tester_objects_are_equal_null() -> None:
    array = pd.array([1, None], dtype="Int64")
    tester = PandasExtensionArrayEqualityTester()
    assert tester.objects_are_equal(array, array.copy(), EqualityConfig())


@pandas_available
def test_pandas_extension_array_equality_tester_objects_are_equal_nan() -> None:
    array = pd.array([1.0, None], dtype="Float64")
    tester = PandasExtensionArrayEqualityTester()
    assert not tester.objects_are_equal(array, array.copy(), EqualityConfig())


@pandas_available
def test_pandas_extension_array_equality_tester_objects_are_equal_null_different() -> None:
    tester = PandasExtensionArrayEqualityTester()
    assert not tester.objects_are_equal(
        pd.array([1, None], dtype="Int64"), pd.array([None, 1], dtype="Int64"), EqualityConfig()
    )


@pandas_available
def test_pandas_extension_array_equality_tester_objects_are_equal_equal_nan() -> None:
    array = pd.array([1, None], dtype="Int64")
    tester = PandasExtensionArrayEqualityTester()
    assert tester.objects_are_equal(array, array.copy(), EqualityConfig(equal_nan=True))


def test_pandas_extension_array_equality_tester_no_pandas() -> None:
    with (
        patch("coola.utils.imports.pandas.is_pandas_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'pandas' package is required but not installed."
        ),
    ):
        PandasExtensionArrayEqualityTester()

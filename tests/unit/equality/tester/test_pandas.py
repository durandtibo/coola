from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig2
from coola.equality.tester import (
    PandasDataFrameEqualityTester,
    PandasSeriesEqualityTester,
)
from coola.testing.fixtures import pandas_available
from coola.utils.imports import is_pandas_available
from tests.unit.equality.utils import ExamplePair

if is_pandas_available():
    import pandas as pd
else:
    pd = Mock()


@pytest.fixture
def config() -> EqualityConfig2:
    return EqualityConfig2()


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
            expected_message="pandas.DataFrames have different elements:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.DataFrame({"col1": [1, 2, 3]}),
            expected=pd.DataFrame({"col2": [1, 2, 3]}),
            expected_message="pandas.DataFrames have different elements:",
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
            expected_message="pandas.Series have different elements:",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1, 2, 3]),
            expected=pd.Series([1, 2, 3, 4]),
            expected_message="pandas.Series have different elements:",
        ),
        id="different shape",
    ),
    pytest.param(
        ExamplePair(
            actual=pd.Series([1, 2, 3]),
            expected=pd.Series([1.0, 2.0, 3.0]),
            expected_message="pandas.Series have different elements:",
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
def test_pandas_dataframe_equality_tester_str() -> None:
    assert str(PandasDataFrameEqualityTester()).startswith("PandasDataFrameEqualityTester(")


@pandas_available
def test_pandas_dataframe_equality_tester_equal_true() -> None:
    assert PandasDataFrameEqualityTester().equal(PandasDataFrameEqualityTester())


@pandas_available
def test_pandas_dataframe_equality_tester__eq__false_different_type() -> None:
    assert not PandasDataFrameEqualityTester().equal(123)


@pandas_available
def test_pandas_dataframe_equality_tester__eq__false_different_type_child() -> None:
    class Child(PandasDataFrameEqualityTester): ...

    assert not PandasDataFrameEqualityTester().equal(Child())


@pandas_available
def test_pandas_dataframe_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig2,
) -> None:
    val = pd.DataFrame({"col": [1, 2, 3]})
    assert PandasDataFrameEqualityTester().objects_are_equal(val, val, config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PandasDataFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PandasDataFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_NOT_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PandasDataFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_NOT_EQUAL)
def test_pandas_dataframe_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PandasDataFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[0].startswith(example.expected_message)


@pandas_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_pandas_dataframe_equality_tester_objects_are_equal_nan(
    config: EqualityConfig2, equal_nan: bool
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
    example: ExamplePair, config: EqualityConfig2
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PandasDataFrameEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pandas_available
def test_pandas_dataframe_equality_tester_no_pandas() -> None:
    with (
        patch("coola.utils.imports.is_pandas_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."),
    ):
        PandasDataFrameEqualityTester()


################################################
#     Tests for PandasSeriesEqualityTester     #
################################################


@pandas_available
def test_pandas_series_equality_tester_str() -> None:
    assert str(PandasSeriesEqualityTester()).startswith("PandasSeriesEqualityTester(")


@pandas_available
def test_pandas_series_equality_tester__eq__true() -> None:
    assert PandasSeriesEqualityTester().equal(PandasSeriesEqualityTester())


@pandas_available
def test_pandas_series_equality_tester__eq__false_different_type() -> None:
    assert not PandasSeriesEqualityTester().equal(123)


@pandas_available
def test_pandas_series_equality_tester__eq__false_different_type_child() -> None:
    class Child(PandasSeriesEqualityTester): ...

    assert not PandasSeriesEqualityTester().equal(Child())


@pandas_available
def test_pandas_series_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig2,
) -> None:
    series = pd.Series([1, 2, 3])
    assert PandasSeriesEqualityTester().objects_are_equal(series, series, config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PandasSeriesEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PandasSeriesEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PandasSeriesEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL)
def test_pandas_series_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PandasSeriesEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[0].startswith(example.expected_message)


@pandas_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_pandas_series_equality_tester_objects_are_equal_nan(
    config: EqualityConfig2, equal_nan: bool
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
    example: ExamplePair, config: EqualityConfig2
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PandasSeriesEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pandas_available
def test_pandas_series_equality_tester_no_pandas() -> None:
    with (
        patch("coola.utils.imports.is_pandas_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."),
    ):
        PandasSeriesEqualityTester()

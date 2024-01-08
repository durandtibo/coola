from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.comparators.pandas_ import (
    PandasDataFrameEqualityComparator,
    PandasSeriesEqualityComparator,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from coola.testing import pandas_available
from coola.utils.imports import is_pandas_available
from tests.unit.equality.comparators.utils import ExamplePair

if is_pandas_available():
    import pandas
else:
    pandas = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


PANDAS_DATAFRAME_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=pandas.DataFrame({}),
            object2=pandas.DataFrame({}),
        ),
        id="0 column",
    ),
    pytest.param(
        ExamplePair(
            object1=pandas.DataFrame({"col": [1, 2, 3]}),
            object2=pandas.DataFrame({"col": [1, 2, 3]}),
        ),
        id="1 column",
    ),
    pytest.param(
        ExamplePair(
            object1=pandas.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            object2=pandas.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
        ),
        id="2 columns",
    ),
]

PANDAS_DATAFRAME_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=pandas.DataFrame({"col": [1, 2, 3]}),
            object2=pandas.DataFrame({"col": [1, 2, 4]}),
            expected_message="pandas.DataFrames have different elements:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            object1=pandas.DataFrame({"col1": [1, 2, 3]}),
            object2=pandas.DataFrame({"col2": [1, 2, 3]}),
            expected_message="pandas.DataFrames have different elements:",
        ),
        id="different column names",
    ),
    pytest.param(
        ExamplePair(
            object1=pandas.DataFrame({"col1": [1, 2, 3]}),
            object2=pandas.Series([1, 2, 3]),
            expected_message="objects have different types:",
        ),
        id="different column names",
    ),
]


PANDAS_SERIES_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=pandas.Series(data=[], dtype=object),
            object2=pandas.Series(data=[], dtype=object),
        ),
        id="empty",
    ),
    pytest.param(
        ExamplePair(object1=pandas.Series([1, 2, 3]), object2=pandas.Series([1, 2, 3])), id="int"
    ),
    pytest.param(
        ExamplePair(object1=pandas.Series(["a", "b", "c"]), object2=pandas.Series(["a", "b", "c"])),
        id="str",
    ),
]

PANDAS_SERIES_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=pandas.Series([1, 2, 3]),
            object2=pandas.Series([1, 2, 4]),
            expected_message="pandas.Series have different elements:",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            object1=pandas.Series([1, 2, 3]),
            object2=pandas.Series([1, 2, 3, 4]),
            expected_message="pandas.Series have different elements:",
        ),
        id="different shape",
    ),
    pytest.param(
        ExamplePair(
            object1=pandas.Series([1, 2, 3]),
            object2=pandas.Series([1.0, 2.0, 3.0]),
            expected_message="pandas.Series have different elements:",
        ),
        id="different data type",
    ),
    pytest.param(
        ExamplePair(
            object1=pandas.Series([1, 2, 3]),
            object2=42,
            expected_message="objects have different types:",
        ),
        id="different type",
    ),
]

#######################################################
#     Tests for PandasDataFrameEqualityComparator     #
#######################################################


@pandas_available
def test_objects_are_equal_dataframe() -> None:
    assert objects_are_equal(
        pandas.DataFrame({"col": [1, 2, 3]}), pandas.DataFrame({"col": [1, 2, 3]})
    )


@pandas_available
def test_pandas_dataframe_equality_comparator_str() -> None:
    assert str(PandasDataFrameEqualityComparator()).startswith("PandasDataFrameEqualityComparator(")


@pandas_available
def test_pandas_dataframe_equality_comparator__eq__true() -> None:
    assert PandasDataFrameEqualityComparator() == PandasDataFrameEqualityComparator()


@pandas_available
def test_pandas_dataframe_equality_comparator__eq__false_different_type() -> None:
    assert PandasDataFrameEqualityComparator() != 123


@pandas_available
def test_pandas_dataframe_equality_comparator_clone() -> None:
    op = PandasDataFrameEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@pandas_available
def test_pandas_dataframe_equality_comparator_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    val = pandas.DataFrame({"col": [1, 2, 3]})
    assert PandasDataFrameEqualityComparator().equal(val, val, config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL)
def test_pandas_dataframe_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL)
def test_pandas_dataframe_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_NOT_EQUAL)
def test_pandas_dataframe_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_NOT_EQUAL)
def test_pandas_dataframe_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[0].startswith(example.expected_message)


@pandas_available
def test_pandas_dataframe_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not PandasDataFrameEqualityComparator().equal(
        object1=pandas.DataFrame({"col": [1, float("nan"), 3]}),
        object2=pandas.DataFrame({"col": [1, float("nan"), 3]}),
        config=config,
    )


@pandas_available
def test_pandas_dataframe_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PandasDataFrameEqualityComparator().equal(
        object1=pandas.DataFrame({"col": [1, float("nan"), 3]}),
        object2=pandas.DataFrame({"col": [1, float("nan"), 3]}),
        config=config,
    )


@pandas_available
def test_pandas_dataframe_equality_comparator_no_pandas() -> None:
    with patch(
        "coola.utils.imports.is_pandas_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`pandas` package is required but not installed."):
        PandasDataFrameEqualityComparator()


####################################################
#     Tests for PandasSeriesEqualityComparator     #
####################################################


@pandas_available
def test_objects_are_equal_series() -> None:
    assert objects_are_equal(
        pandas.DataFrame({"col": [1, 2, 3]}), pandas.DataFrame({"col": [1, 2, 3]})
    )


@pandas_available
def test_pandas_series_equality_comparator_str() -> None:
    assert str(PandasSeriesEqualityComparator()).startswith("PandasSeriesEqualityComparator(")


@pandas_available
def test_pandas_series_equality_comparator__eq__true() -> None:
    assert PandasSeriesEqualityComparator() == PandasSeriesEqualityComparator()


@pandas_available
def test_pandas_series_equality_comparator__eq__false_different_type() -> None:
    assert PandasSeriesEqualityComparator() != 123


@pandas_available
def test_pandas_series_equality_comparator_clone() -> None:
    op = PandasSeriesEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@pandas_available
def test_pandas_series_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    series = pandas.Series([1, 2, 3])
    assert PandasSeriesEqualityComparator().equal(series, series, config)


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL)
def test_pandas_series_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL)
def test_pandas_series_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL)
def test_pandas_series_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_NOT_EQUAL)
def test_pandas_series_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[0].startswith(example.expected_message)


@pandas_available
def test_pandas_series_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not PandasSeriesEqualityComparator().equal(
        object1=pandas.Series([0.0, float("nan"), float("nan"), 1.2]),
        object2=pandas.Series([0.0, float("nan"), float("nan"), 1.2]),
        config=config,
    )


@pandas_available
def test_pandas_series_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PandasSeriesEqualityComparator().equal(
        object1=pandas.Series([0.0, float("nan"), float("nan"), 1.2]),
        object2=pandas.Series([0.0, float("nan"), float("nan"), 1.2]),
        config=config,
    )


@pandas_available
def test_pandas_series_equality_comparator_no_pandas() -> None:
    with patch(
        "coola.utils.imports.is_pandas_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`pandas` package is required but not installed."):
        PandasSeriesEqualityComparator()


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


@pandas_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        pandas.DataFrame: PandasDataFrameEqualityComparator(),
        pandas.Series: PandasSeriesEqualityComparator(),
    }


def test_get_type_comparator_mapping_no_pandas() -> None:
    with patch(
        "coola.equality.comparators.pandas_.is_pandas_available", lambda *args, **kwargs: False
    ):
        assert get_type_comparator_mapping() == {}

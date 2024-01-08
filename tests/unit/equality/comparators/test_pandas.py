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

if is_pandas_available():
    import pandas
else:
    pandas = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


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
def test_pandas_dataframe_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=pandas.DataFrame({"col": [1, 2, 3]}),
            object2=pandas.DataFrame({"col": [1, 2, 3]}),
            config=config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=pandas.DataFrame({"col": [1, 2, 3]}),
            object2=pandas.DataFrame({"col": [1, 2, 3]}),
            config=config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equality_comparator_equal_false_different_value(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=pandas.DataFrame({"col": [1, 2, 3]}),
            object2=pandas.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=pandas.DataFrame({"col": [1, 2, 3]}),
            object2=pandas.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("pandas.DataFrames have different elements:")


@pandas_available
def test_pandas_dataframe_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=pandas.DataFrame({"col": [1, 2, 3]}), object2=42, config=config
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PandasDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=pandas.DataFrame({"col": [1, 2, 3]}), object2=42, config=config
        )
        assert caplog.messages[0].startswith("objects have different types:")


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
def test_pandas_series_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=pandas.Series([1, 2, 3]),
            object2=pandas.Series([1, 2, 3]),
            config=config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_series_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=pandas.Series([1, 2, 3]),
            object2=pandas.Series([1, 2, 3]),
            config=config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_series_equality_comparator_equal_false_different_value(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=pandas.Series([1, 2, 3]), object2=pandas.Series([1, 2, 4]), config=config
        )
        assert not caplog.messages


@pandas_available
def test_pandas_series_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=pandas.Series([1, 2, 3]), object2=pandas.Series([1, 2, 4]), config=config
        )
        assert caplog.messages[0].startswith("pandas.Series have different elements:")


@pandas_available
def test_pandas_series_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=pandas.Series([1, 2, 3]), object2=42, config=config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PandasSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=pandas.Series([1, 2, 3]), object2=42, config=config)
        assert caplog.messages[0].startswith("objects have different types:")


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

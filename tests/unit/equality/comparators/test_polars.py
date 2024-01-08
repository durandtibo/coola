from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.comparators.polars_ import (
    PolarsDataFrameEqualityComparator,
    PolarsSeriesEqualityComparator,
    get_type_comparator_mapping,
)
from coola.testers import EqualityTester
from coola.testing import polars_available
from coola.utils.imports import is_polars_available

if is_polars_available():
    import polars
else:
    polars = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#######################################################
#     Tests for PolarsDataFrameEqualityComparator     #
#######################################################


@polars_available
def test_objects_are_equal_dataframe() -> None:
    assert objects_are_equal(
        polars.DataFrame({"col": [1, 2, 3]}), polars.DataFrame({"col": [1, 2, 3]})
    )


@polars_available
def test_polars_dataframe_equality_comparator_str() -> None:
    assert str(PolarsDataFrameEqualityComparator()).startswith("PolarsDataFrameEqualityComparator(")


@polars_available
def test_polars_dataframe_equality_comparator__eq__true() -> None:
    assert PolarsDataFrameEqualityComparator() == PolarsDataFrameEqualityComparator()


@polars_available
def test_polars_dataframe_equality_comparator__eq__false_different_type() -> None:
    assert PolarsDataFrameEqualityComparator() != 123


@polars_available
def test_polars_dataframe_equality_comparator_clone() -> None:
    op = PolarsDataFrameEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@polars_available
def test_polars_dataframe_equality_comparator_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    val = polars.DataFrame({"col": [1, 2, 3]})
    assert PolarsDataFrameEqualityComparator().equal(val, val, config)


@polars_available
def test_polars_dataframe_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=polars.DataFrame({"col": [1, 2, 3]}),
            object2=polars.DataFrame({"col": [1, 2, 3]}),
            config=config,
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=polars.DataFrame({"col": [1, 2, 3]}),
            object2=polars.DataFrame({"col": [1, 2, 3]}),
            config=config,
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equality_comparator_equal_false_different_value(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=polars.DataFrame({"col": [1, 2, 3]}),
            object2=polars.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=polars.DataFrame({"col": [1, 2, 3]}),
            object2=polars.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.DataFrames have different elements:")


@polars_available
def test_polars_dataframe_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=polars.DataFrame({"col": [1, 2, 3]}), object2=42, config=config
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=polars.DataFrame({"col": [1, 2, 3]}), object2=42, config=config
        )
        assert caplog.messages[0].startswith("objects have different types:")


@polars_available
def test_polars_dataframe_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not PolarsDataFrameEqualityComparator().equal(
        object1=polars.DataFrame({"col": [1, float("nan"), 3]}),
        object2=polars.DataFrame({"col": [1, float("nan"), 3]}),
        config=config,
    )


@polars_available
def test_polars_dataframe_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PolarsDataFrameEqualityComparator().equal(
        object1=polars.DataFrame({"col": [1, float("nan"), 3]}),
        object2=polars.DataFrame({"col": [1, float("nan"), 3]}),
        config=config,
    )


@polars_available
def test_polars_dataframe_equality_comparator_no_polars() -> None:
    with patch(
        "coola.utils.imports.is_polars_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`polars` package is required but not installed."):
        PolarsDataFrameEqualityComparator()


####################################################
#     Tests for PolarsSeriesEqualityComparator     #
####################################################


@polars_available
def test_objects_are_equal_series() -> None:
    assert objects_are_equal(
        polars.DataFrame({"col": [1, 2, 3]}), polars.DataFrame({"col": [1, 2, 3]})
    )


@polars_available
def test_polars_series_equality_comparator_str() -> None:
    assert str(PolarsSeriesEqualityComparator()).startswith("PolarsSeriesEqualityComparator(")


@polars_available
def test_polars_series_equality_comparator__eq__true() -> None:
    assert PolarsSeriesEqualityComparator() == PolarsSeriesEqualityComparator()


@polars_available
def test_polars_series_equality_comparator__eq__false_different_type() -> None:
    assert PolarsSeriesEqualityComparator() != 123


@polars_available
def test_polars_series_equality_comparator_clone() -> None:
    op = PolarsSeriesEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@polars_available
def test_polars_series_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    series = polars.Series([1, 2, 3])
    assert PolarsSeriesEqualityComparator().equal(series, series, config)


@polars_available
def test_polars_series_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=polars.Series([1, 2, 3]),
            object2=polars.Series([1, 2, 3]),
            config=config,
        )
        assert not caplog.messages


@polars_available
def test_polars_series_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=polars.Series([1, 2, 3]),
            object2=polars.Series([1, 2, 3]),
            config=config,
        )
        assert not caplog.messages


@polars_available
def test_polars_series_equality_comparator_equal_false_different_value(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=polars.Series([1, 2, 3]), object2=polars.Series([1, 2, 4]), config=config
        )
        assert not caplog.messages


@polars_available
def test_polars_series_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=polars.Series([1, 2, 3]), object2=polars.Series([1, 2, 4]), config=config
        )
        assert caplog.messages[0].startswith("polars.Series have different elements:")


@polars_available
def test_polars_series_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=polars.Series([1, 2, 3]), object2=42, config=config)
        assert not caplog.messages


@polars_available
def test_polars_series_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=polars.Series([1, 2, 3]), object2=42, config=config)
        assert caplog.messages[0].startswith("objects have different types:")


@polars_available
def test_polars_series_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not PolarsSeriesEqualityComparator().equal(
        object1=polars.Series([0.0, float("nan"), float("nan"), 1.2]),
        object2=polars.Series([0.0, float("nan"), float("nan"), 1.2]),
        config=config,
    )


@polars_available
def test_polars_series_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PolarsSeriesEqualityComparator().equal(
        object1=polars.Series([0.0, float("nan"), float("nan"), 1.2]),
        object2=polars.Series([0.0, float("nan"), float("nan"), 1.2]),
        config=config,
    )


@polars_available
def test_polars_series_equality_comparator_no_polars() -> None:
    with patch(
        "coola.utils.imports.is_polars_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`polars` package is required but not installed."):
        PolarsSeriesEqualityComparator()


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


@polars_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        polars.DataFrame: PolarsDataFrameEqualityComparator(),
        polars.Series: PolarsSeriesEqualityComparator(),
    }


def test_get_type_comparator_mapping_no_polars() -> None:
    with patch(
        "coola.equality.comparators.polars_.is_polars_available", lambda *args, **kwargs: False
    ):
        assert get_type_comparator_mapping() == {}

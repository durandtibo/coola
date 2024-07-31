from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators.polars_ import (
    PolarsDataFrameEqualityComparator,
    PolarsSeriesEqualityComparator,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from coola.testing import polars_available
from coola.utils.imports import is_polars_available
from tests.unit.equality.comparators.utils import ExamplePair

if is_polars_available():
    import polars as pl
else:
    pl = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


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
            expected_message="polars.DataFrames have different elements:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.DataFrame({"col1": [1, 2, 3]}),
            expected=pl.DataFrame({"col2": [1, 2, 3]}),
            expected_message="polars.DataFrames have different elements:",
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
            expected_message="polars.Series have different elements:",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1, 2, 3]),
            expected=pl.Series([1, 2, 3, 4]),
            expected_message="polars.Series have different elements:",
        ),
        id="different shape",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.Series([1, 2, 3]),
            expected=pl.Series([1.0, 2.0, 3.0]),
            expected_message="polars.Series have different elements:",
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

POLARS_EQUAL = POLARS_SERIES_EQUAL + POLARS_DATAFRAME_EQUAL
POLARS_NOT_EQUAL = POLARS_SERIES_NOT_EQUAL + POLARS_DATAFRAME_NOT_EQUAL
POLARS_EQUAL_TOLERANCE = POLARS_SERIES_EQUAL_TOLERANCE + POLARS_DATAFRAME_EQUAL_TOLERANCE

#######################################################
#     Tests for PolarsDataFrameEqualityComparator     #
#######################################################


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
    val = pl.DataFrame({"col": [1, 2, 3]})
    assert PolarsDataFrameEqualityComparator().equal(val, val, config)


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_EQUAL)
def test_polars_dataframe_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_EQUAL)
def test_polars_dataframe_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_NOT_EQUAL)
def test_polars_dataframe_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_NOT_EQUAL)
def test_polars_dataframe_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = PolarsDataFrameEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[0].startswith(example.expected_message)


@polars_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_polars_dataframe_equality_comparator_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        PolarsDataFrameEqualityComparator().equal(
            actual=pl.DataFrame({"col": [1.0, float("nan"), 3.0]}),
            expected=pl.DataFrame({"col": [1.0, float("nan"), 3.0]}),
            config=config,
        )
        == equal_nan
    )


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_EQUAL_TOLERANCE)
def test_polars_dataframe_equality_comparator_equal_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PolarsDataFrameEqualityComparator().equal(
        actual=example.actual, expected=example.expected, config=config
    )


@polars_available
def test_polars_dataframe_equality_comparator_no_polars() -> None:
    with (
        patch("coola.utils.imports.is_polars_available", lambda: False),
        pytest.raises(RuntimeError, match="'polars' package is required but not installed."),
    ):
        PolarsDataFrameEqualityComparator()


####################################################
#     Tests for PolarsSeriesEqualityComparator     #
####################################################


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
    series = pl.Series([1, 2, 3])
    assert PolarsSeriesEqualityComparator().equal(series, series, config)


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_EQUAL)
def test_polars_series_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_EQUAL)
def test_polars_series_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_NOT_EQUAL)
def test_polars_series_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_NOT_EQUAL)
def test_polars_series_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = PolarsSeriesEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[0].startswith(example.expected_message)


@polars_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_polars_series_equality_comparator_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        PolarsSeriesEqualityComparator().equal(
            actual=pl.Series([0.0, float("nan"), float("nan"), 1.2]),
            expected=pl.Series([0.0, float("nan"), float("nan"), 1.2]),
            config=config,
        )
        == equal_nan
    )


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_EQUAL_TOLERANCE)
def test_polars_series_equality_comparator_equal_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PolarsSeriesEqualityComparator().equal(
        actual=example.actual, expected=example.expected, config=config
    )


@polars_available
def test_polars_series_equality_comparator_no_polars() -> None:
    with (
        patch("coola.utils.imports.is_polars_available", lambda: False),
        pytest.raises(RuntimeError, match="'polars' package is required but not installed."),
    ):
        PolarsSeriesEqualityComparator()


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


@polars_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        pl.DataFrame: PolarsDataFrameEqualityComparator(),
        pl.Series: PolarsSeriesEqualityComparator(),
    }


def test_get_type_comparator_mapping_no_polars() -> None:
    with patch("coola.equality.comparators.polars_.is_polars_available", lambda: False):
        assert get_type_comparator_mapping() == {}

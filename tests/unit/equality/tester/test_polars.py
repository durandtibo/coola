from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    PolarsDataFrameEqualityTester,
    PolarsLazyFrameEqualityTester,
    PolarsSeriesEqualityTester,
)
from coola.testing.fixtures import polars_available
from coola.utils.imports import is_polars_available
from tests.unit.equality.utils import ExamplePair

if is_polars_available():
    import polars as pl
else:
    pl = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


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
            expected_message="polars.LazyFrames have different elements:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pl.LazyFrame({"col1": [1, 2, 3]}),
            expected=pl.LazyFrame({"col2": [1, 2, 3]}),
            expected_message="polars.LazyFrames have different elements:",
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

POLARS_EQUAL = POLARS_SERIES_EQUAL + POLARS_DATAFRAME_EQUAL + POLARS_LAZYFRAME_EQUAL
POLARS_NOT_EQUAL = POLARS_SERIES_NOT_EQUAL + POLARS_DATAFRAME_NOT_EQUAL + POLARS_LAZYFRAME_NOT_EQUAL
POLARS_EQUAL_TOLERANCE = (
    POLARS_SERIES_EQUAL_TOLERANCE
    + POLARS_DATAFRAME_EQUAL_TOLERANCE
    + POLARS_LAZYFRAME_EQUAL_TOLERANCE
)


###################################################
#     Tests for PolarsDataFrameEqualityTester     #
###################################################


@polars_available
def test_polars_dataframe_equality_tester_repr() -> None:
    assert repr(PolarsDataFrameEqualityTester()).startswith("PolarsDataFrameEqualityTester(")


@polars_available
def test_polars_dataframe_equality_tester_str() -> None:
    assert str(PolarsDataFrameEqualityTester()).startswith("PolarsDataFrameEqualityTester(")


@polars_available
def test_polars_dataframe_equality_tester_equal_true() -> None:
    assert PolarsDataFrameEqualityTester().equal(PolarsDataFrameEqualityTester())


@polars_available
def test_polars_dataframe_equality_tester_equal_false_different_type() -> None:
    assert not PolarsDataFrameEqualityTester().equal(123)


@polars_available
def test_polars_dataframe_equality_tester_equal_false_different_type_child() -> None:
    class Child(PolarsDataFrameEqualityTester): ...

    assert not PolarsDataFrameEqualityTester().equal(Child())


@polars_available
def test_polars_dataframe_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    val = pl.DataFrame({"col": [1, 2, 3]})
    assert PolarsDataFrameEqualityTester().objects_are_equal(val, val, config)


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_EQUAL)
def test_polars_dataframe_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PolarsDataFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_EQUAL)
def test_polars_dataframe_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PolarsDataFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_NOT_EQUAL)
def test_polars_dataframe_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PolarsDataFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_NOT_EQUAL)
def test_polars_dataframe_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PolarsDataFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[0].startswith(example.expected_message)


@polars_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_polars_dataframe_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        PolarsDataFrameEqualityTester().objects_are_equal(
            actual=pl.DataFrame({"col": [1.0, float("nan"), 3.0]}),
            expected=pl.DataFrame({"col": [1.0, float("nan"), 3.0]}),
            config=config,
        )
        == equal_nan
    )


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_EQUAL_TOLERANCE)
def test_polars_dataframe_equality_tester_objects_are_equal_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PolarsDataFrameEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


def test_polars_dataframe_equality_tester_no_polars() -> None:
    with (
        patch("coola.utils.imports.is_polars_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."),
    ):
        PolarsDataFrameEqualityTester()


###################################################
#     Tests for PolarsLazyFrameEqualityTester     #
###################################################


@polars_available
def test_polars_lazyframe_equality_tester_repr() -> None:
    assert repr(PolarsLazyFrameEqualityTester()).startswith("PolarsLazyFrameEqualityTester(")


@polars_available
def test_polars_lazyframe_equality_tester_str() -> None:
    assert str(PolarsLazyFrameEqualityTester()).startswith("PolarsLazyFrameEqualityTester(")


@polars_available
def test_polars_lazyframe_equality_tester_equal_true() -> None:
    assert PolarsLazyFrameEqualityTester().equal(PolarsLazyFrameEqualityTester())


@polars_available
def test_polars_lazyframe_equality_tester_equal_false_different_type() -> None:
    assert not PolarsLazyFrameEqualityTester().equal(123)


@polars_available
def test_polars_lazyframe_equality_tester_equal_false_different_type_child() -> None:
    class Child(PolarsLazyFrameEqualityTester): ...

    assert not PolarsLazyFrameEqualityTester().equal(Child())


@polars_available
def test_polars_lazyframe_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    val = pl.LazyFrame({"col": [1, 2, 3]})
    assert PolarsLazyFrameEqualityTester().objects_are_equal(val, val, config)


@polars_available
@pytest.mark.parametrize("example", POLARS_LAZYFRAME_EQUAL)
def test_polars_lazyframe_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PolarsLazyFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_LAZYFRAME_EQUAL)
def test_polars_lazyframe_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PolarsLazyFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_LAZYFRAME_NOT_EQUAL)
def test_polars_lazyframe_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PolarsLazyFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_LAZYFRAME_NOT_EQUAL)
def test_polars_lazyframe_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PolarsLazyFrameEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[0].startswith(example.expected_message)


@polars_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_polars_lazyframe_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        PolarsLazyFrameEqualityTester().objects_are_equal(
            actual=pl.LazyFrame({"col": [1.0, float("nan"), 3.0]}),
            expected=pl.LazyFrame({"col": [1.0, float("nan"), 3.0]}),
            config=config,
        )
        == equal_nan
    )


@polars_available
@pytest.mark.parametrize("example", POLARS_LAZYFRAME_EQUAL_TOLERANCE)
def test_polars_lazyframe_equality_tester_objects_are_equal_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PolarsLazyFrameEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


def test_polars_lazyframe_equality_tester_no_polars() -> None:
    with (
        patch("coola.utils.imports.is_polars_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."),
    ):
        PolarsLazyFrameEqualityTester()


################################################
#     Tests for PolarsSeriesEqualityTester     #
################################################


@polars_available
def test_polars_series_equality_tester_repr() -> None:
    assert repr(PolarsSeriesEqualityTester()).startswith("PolarsSeriesEqualityTester(")


@polars_available
def test_polars_series_equality_tester_str() -> None:
    assert str(PolarsSeriesEqualityTester()).startswith("PolarsSeriesEqualityTester(")


@polars_available
def test_polars_series_equality_tester_equal_true() -> None:
    assert PolarsSeriesEqualityTester().equal(PolarsSeriesEqualityTester())


@polars_available
def test_polars_series_equality_tester_equal_false_different_type() -> None:
    assert not PolarsSeriesEqualityTester().equal(123)


@polars_available
def test_polars_series_equality_tester_equal_false_different_type_child() -> None:
    class Child(PolarsSeriesEqualityTester): ...

    assert not PolarsSeriesEqualityTester().equal(Child())


@polars_available
def test_polars_series_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    series = pl.Series([1, 2, 3])
    assert PolarsSeriesEqualityTester().objects_are_equal(series, series, config)


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_EQUAL)
def test_polars_series_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PolarsSeriesEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_EQUAL)
def test_polars_series_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PolarsSeriesEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_NOT_EQUAL)
def test_polars_series_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PolarsSeriesEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_NOT_EQUAL)
def test_polars_series_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PolarsSeriesEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[0].startswith(example.expected_message)


@polars_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_polars_series_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        PolarsSeriesEqualityTester().objects_are_equal(
            actual=pl.Series([0.0, float("nan"), float("nan"), 1.2]),
            expected=pl.Series([0.0, float("nan"), float("nan"), 1.2]),
            config=config,
        )
        == equal_nan
    )


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_EQUAL_TOLERANCE)
def test_polars_series_equality_tester_objects_are_equal_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PolarsSeriesEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


def test_polars_series_equality_tester_no_polars() -> None:
    with (
        patch("coola.utils.imports.is_polars_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."),
    ):
        PolarsSeriesEqualityTester()

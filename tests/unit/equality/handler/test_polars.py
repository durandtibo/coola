from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    PolarsDataFrameEqualHandler,
    PolarsSeriesEqualHandler,
)
from coola.equality.handler.polars import (
    PolarsLazyFrameEqualHandler,
    assert_frame_equal,
    assert_series_equal,
    has_nan,
    is_new_naming,
)
from coola.equality.testers import EqualityTester
from coola.testing.fixtures import polars_available
from coola.utils.imports import is_polars_available
from tests.unit.equality.comparators.test_polars import (
    POLARS_DATAFRAME_EQUAL_TOLERANCE,
    POLARS_LAZYFRAME_EQUAL_TOLERANCE,
    POLARS_SERIES_EQUAL_TOLERANCE,
)

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    pl = Mock()

if TYPE_CHECKING:
    from tests.unit.equality.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#################################################
#     Tests for PolarsDataFrameEqualHandler     #
#################################################


def test_polars_dataframe_equal_handler__eq__true() -> None:
    assert PolarsDataFrameEqualHandler() == PolarsDataFrameEqualHandler()


def test_polars_dataframe_equal_handler__eq__false_different_type() -> None:
    assert PolarsDataFrameEqualHandler() != FalseHandler()


def test_polars_dataframe_equal_handler__eq__false_different_type_child() -> None:
    class Child(PolarsDataFrameEqualHandler): ...

    assert PolarsDataFrameEqualHandler() != Child()


def test_polars_dataframe_equal_handler_repr() -> None:
    assert repr(PolarsDataFrameEqualHandler()).startswith("PolarsDataFrameEqualHandler(")


def test_polars_dataframe_equal_handler_str() -> None:
    assert str(PolarsDataFrameEqualHandler()).startswith("PolarsDataFrameEqualHandler(")


@polars_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pl.DataFrame({}), pl.DataFrame({})),
        (pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 3]})),
        (
            pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
        ),
    ],
)
def test_polars_dataframe_equal_handler_handle_true(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(actual, expected, config)
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(
            pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 3]}), config
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(pl.DataFrame({}), pl.DataFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_column(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.DataFrame({"col1": [1, 2, 3]}), pl.DataFrame({"col2": [1, 2, 3]}), config
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_value(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 4]}), config
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.DataFrame(data={"col": [1, 2, 3]}),
            pl.DataFrame(data={"col": [1.0, 2.0, 3.0]}),
            config,
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.DataFrame({"col": [1, 2, 3]}),
            pl.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.DataFrames have different elements:")


@polars_available
def test_polars_dataframe_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PolarsDataFrameEqualHandler().handle(
        pl.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        pl.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@polars_available
def test_polars_dataframe_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PolarsDataFrameEqualHandler().handle(
        pl.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        pl.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@polars_available
@pytest.mark.parametrize("example", POLARS_DATAFRAME_EQUAL_TOLERANCE)
def test_polars_dataframe_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PolarsDataFrameEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_polars_dataframe_equal_handler_set_next_handler() -> None:
    PolarsDataFrameEqualHandler().set_next_handler(FalseHandler())


#################################################
#     Tests for PolarsLazyFrameEqualHandler     #
#################################################


def test_polars_lazyframe_equal_handler__eq__true() -> None:
    assert PolarsLazyFrameEqualHandler() == PolarsLazyFrameEqualHandler()


def test_polars_lazyframe_equal_handler__eq__false_different_type() -> None:
    assert PolarsLazyFrameEqualHandler() != FalseHandler()


def test_polars_lazyframe_equal_handler__eq__false_different_type_child() -> None:
    class Child(PolarsLazyFrameEqualHandler): ...

    assert PolarsLazyFrameEqualHandler() != Child()


def test_polars_lazyframe_equal_handler_repr() -> None:
    assert repr(PolarsLazyFrameEqualHandler()).startswith("PolarsLazyFrameEqualHandler(")


def test_polars_lazyframe_equal_handler_str() -> None:
    assert str(PolarsLazyFrameEqualHandler()).startswith("PolarsLazyFrameEqualHandler(")


@polars_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pl.LazyFrame({}), pl.LazyFrame({})),
        (pl.LazyFrame({"col": [1, 2, 3]}), pl.LazyFrame({"col": [1, 2, 3]})),
        (
            pl.LazyFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            pl.LazyFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
        ),
    ],
)
def test_polars_lazyframe_equal_handler_handle_true(
    actual: pl.LazyFrame,
    expected: pl.LazyFrame,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(actual, expected, config)
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(
            pl.LazyFrame({"col": [1, 2, 3]}), pl.LazyFrame({"col": [1, 2, 3]}), config
        )
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(pl.LazyFrame({}), pl.LazyFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_column(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.LazyFrame({"col1": [1, 2, 3]}), pl.LazyFrame({"col2": [1, 2, 3]}), config
        )
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_value(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.LazyFrame({"col": [1, 2, 3]}), pl.LazyFrame({"col": [1, 2, 4]}), config
        )
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.LazyFrame(data={"col": [1, 2, 3]}),
            pl.LazyFrame(data={"col": [1.0, 2.0, 3.0]}),
            config,
        )
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.LazyFrame({"col": [1, 2, 3]}),
            pl.LazyFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.LazyFrames have different elements:")


@polars_available
def test_polars_lazyframe_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PolarsLazyFrameEqualHandler().handle(
        pl.LazyFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        pl.LazyFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@polars_available
def test_polars_lazyframe_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PolarsLazyFrameEqualHandler().handle(
        pl.LazyFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        pl.LazyFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@polars_available
@pytest.mark.parametrize("example", POLARS_LAZYFRAME_EQUAL_TOLERANCE)
def test_polars_lazyframe_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PolarsLazyFrameEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_polars_lazyframe_equal_handler_set_next_handler() -> None:
    PolarsLazyFrameEqualHandler().set_next_handler(FalseHandler())


##############################################
#     Tests for PolarsSeriesEqualHandler     #
##############################################


def test_polars_series_equal_handler__eq__true() -> None:
    assert PolarsSeriesEqualHandler() == PolarsSeriesEqualHandler()


def test_polars_series_equal_handler__eq__false_different_type() -> None:
    assert PolarsSeriesEqualHandler() != FalseHandler()


def test_polars_series_equal_handler__eq__false_different_type_child() -> None:
    class Child(PolarsSeriesEqualHandler): ...

    assert PolarsSeriesEqualHandler() != Child()


def test_polars_series_equal_handler_repr() -> None:
    assert repr(PolarsSeriesEqualHandler()).startswith("PolarsSeriesEqualHandler(")


def test_polars_series_equal_handler_str() -> None:
    assert str(PolarsSeriesEqualHandler()).startswith("PolarsSeriesEqualHandler(")


@polars_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pl.Series([]), pl.Series([])),
        (pl.Series([1, 2, 3]), pl.Series([1, 2, 3])),
        (pl.Series(["a", "b", "c"]), pl.Series(["a", "b", "c"])),
    ],
)
def test_polars_series_equal_handler_handle_true(
    actual: pl.Series,
    expected: pl.Series,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(actual, expected, config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_different_shape(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 3, 4]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pl.Series([1, 2, 3]),
            pl.Series([1.0, 2.0, 3.0]),
            config,
        )
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_different_value(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=pl.Series([1, 2, 3]),
            expected=pl.Series([1, 2, 3, 4]),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.Series have different elements:")


@polars_available
def test_polars_series_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PolarsSeriesEqualHandler().handle(
        pl.Series([0.0, float("nan"), float("nan"), 1.2]),
        pl.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@polars_available
def test_polars_series_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PolarsSeriesEqualHandler().handle(
        pl.Series([0.0, float("nan"), float("nan"), 1.2]),
        pl.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@polars_available
@pytest.mark.parametrize("example", POLARS_SERIES_EQUAL_TOLERANCE)
def test_polars_series_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PolarsSeriesEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_polars_series_equal_handler_set_next_handler() -> None:
    PolarsSeriesEqualHandler().set_next_handler(FalseHandler())


#############################
#     Tests for has_nan     #
#############################

HAS_NAN_TRUE = [
    pl.Series(["A", "B", "C"]),
    pl.Series([1, 2, 3]),
    pl.Series([1, 2, 3], dtype=pl.Int64),
    pl.Series([1, 2, 3], dtype=pl.Float64),
    pl.Series([True]),
    pl.Series([], dtype=pl.Float32),
    pl.DataFrame({"col": [1, 2, 3]}),
    pl.DataFrame({"col": [1, 2, 3]}, schema={"col": pl.Int64}),
    pl.DataFrame({"col": [1, 2, 3]}, schema={"col": pl.Float64}),
    pl.DataFrame({"col": ["A", "B", "C"]}),
    pl.DataFrame({}),
]
HAS_NAN_FALSE = [
    pl.Series([1.0, 2.0, float("nan")]),
    pl.Series([1.0, float("nan"), 3.0], dtype=pl.Float64),
    pl.DataFrame({"col": [1.0, 2.0, float("nan")]}),
    pl.DataFrame({"col": [1.0, 2.0, float("nan")]}, schema={"col": pl.Float64}),
]


@polars_available
@pytest.mark.parametrize("df_or_series", HAS_NAN_TRUE)
def test_has_nan_true(df_or_series: pl.DataFrame | pl.Series) -> None:
    assert not has_nan(df_or_series)


@polars_available
@pytest.mark.parametrize("df_or_series", HAS_NAN_FALSE)
def test_has_nan_false(df_or_series: pl.DataFrame | pl.Series) -> None:
    assert has_nan(df_or_series)


########################################
#     Tests for assert_frame_equal     #
########################################


@polars_available
def test_assert_frame_equal_new_naming() -> None:
    is_new_naming.cache_clear()
    with (
        patch("coola.equality.handler.polars.is_new_naming", lambda: True),
        patch("polars.testing.assert_frame_equal") as mock,
    ):
        assert_frame_equal(
            df1=pl.DataFrame({"col": [1, 2, 3]}), df2=pl.DataFrame({"col": [1, 2, 3]})
        )
        assert mock.call_args.kwargs == {"abs_tol": 1e-8, "rel_tol": 1e-5, "check_exact": False}


@polars_available
def test_assert_frame_equal_old_naming() -> None:
    is_new_naming.cache_clear()
    with (
        patch("coola.equality.handler.polars.is_new_naming", lambda: False),
        patch("polars.testing.assert_frame_equal") as mock,
    ):
        assert_frame_equal(
            df1=pl.DataFrame({"col": [1, 2, 3]}), df2=pl.DataFrame({"col": [1, 2, 3]})
        )
        assert mock.call_args.kwargs == {"atol": 1e-8, "rtol": 1e-5, "check_exact": False}


#########################################
#     Tests for assert_series_equal     #
#########################################


@polars_available
def test_assert_series_equal_new_naming() -> None:
    is_new_naming.cache_clear()
    with (
        patch("coola.equality.handler.polars.is_new_naming", lambda: True),
        patch("polars.testing.assert_series_equal") as mock,
    ):
        assert_series_equal(series1=pl.Series([1, 2, 3]), series2=pl.Series([1, 2, 3]))
        assert mock.call_args.kwargs == {"abs_tol": 1e-8, "rel_tol": 1e-5, "check_exact": False}


@polars_available
def test_assert_series_equal_old_naming() -> None:
    is_new_naming.cache_clear()
    with (
        patch("coola.equality.handler.polars.is_new_naming", lambda: False),
        patch("polars.testing.assert_series_equal") as mock,
    ):
        assert_series_equal(series1=pl.Series([1, 2, 3]), series2=pl.Series([1, 2, 3]))
        assert mock.call_args.kwargs == {"atol": 1e-8, "rtol": 1e-5, "check_exact": False}

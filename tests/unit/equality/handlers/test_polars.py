from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import (
    FalseHandler,
    PolarsDataFrameEqualHandler,
    PolarsSeriesEqualHandler,
)
from coola.equality.handlers.polars_ import has_nan
from coola.equality.testers import EqualityTester
from coola.testing import polars_available
from coola.utils import is_polars_available
from tests.unit.equality.comparators.test_polars import (
    POLARS_DATAFRAME_EQUAL_TOLERANCE,
    POLARS_SERIES_EQUAL_TOLERANCE,
)

if is_polars_available():
    import polars
else:  # pragma: no cover
    polars = Mock()

if TYPE_CHECKING:
    from tests.unit.equality.comparators.utils import ExamplePair


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#################################################
#     Tests for PolarsDataFrameEqualHandler     #
#################################################


def test_polars_dataframe_equal_handler_eq_true() -> None:
    assert PolarsDataFrameEqualHandler() == PolarsDataFrameEqualHandler()


def test_polars_dataframe_equal_handler_eq_false() -> None:
    assert PolarsDataFrameEqualHandler() != FalseHandler()


def test_polars_dataframe_equal_handler_repr() -> None:
    assert repr(PolarsDataFrameEqualHandler()).startswith("PolarsDataFrameEqualHandler(")


def test_polars_dataframe_equal_handler_str() -> None:
    assert str(PolarsDataFrameEqualHandler()).startswith("PolarsDataFrameEqualHandler(")


@polars_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (polars.DataFrame({}), polars.DataFrame({})),
        (polars.DataFrame({"col": [1, 2, 3]}), polars.DataFrame({"col": [1, 2, 3]})),
        (
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            polars.DataFrame(
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
    actual: polars.DataFrame,
    expected: polars.DataFrame,
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
            polars.DataFrame({"col": [1, 2, 3]}), polars.DataFrame({"col": [1, 2, 3]}), config
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            polars.DataFrame({}), polars.DataFrame({"col": [1, 2, 3]}), config
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_column(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            polars.DataFrame({"col1": [1, 2, 3]}), polars.DataFrame({"col2": [1, 2, 3]}), config
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
            polars.DataFrame({"col": [1, 2, 3]}), polars.DataFrame({"col": [1, 2, 4]}), config
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
            polars.DataFrame(data={"col": [1, 2, 3]}),
            polars.DataFrame(data={"col": [1.0, 2.0, 3.0]}),
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
            polars.DataFrame({"col": [1, 2, 3]}),
            polars.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.DataFrames have different elements:")


@polars_available
def test_polars_dataframe_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PolarsDataFrameEqualHandler().handle(
        polars.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        polars.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@polars_available
def test_polars_dataframe_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PolarsDataFrameEqualHandler().handle(
        polars.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        polars.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
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


##############################################
#     Tests for PolarsSeriesEqualHandler     #
##############################################


def test_polars_series_equal_handler_eq_true() -> None:
    assert PolarsSeriesEqualHandler() == PolarsSeriesEqualHandler()


def test_polars_series_equal_handler_eq_false() -> None:
    assert PolarsSeriesEqualHandler() != FalseHandler()


def test_polars_series_equal_handler_repr() -> None:
    assert repr(PolarsSeriesEqualHandler()).startswith("PolarsSeriesEqualHandler(")


def test_polars_series_equal_handler_str() -> None:
    assert str(PolarsSeriesEqualHandler()).startswith("PolarsSeriesEqualHandler(")


@polars_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (polars.Series([]), polars.Series([])),
        (polars.Series([1, 2, 3]), polars.Series([1, 2, 3])),
        (polars.Series(["a", "b", "c"]), polars.Series(["a", "b", "c"])),
    ],
)
def test_polars_series_equal_handler_handle_true(
    actual: polars.Series,
    expected: polars.Series,
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
        assert handler.handle(polars.Series([1, 2, 3]), polars.Series([1, 2, 3]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_different_shape(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(polars.Series([1, 2, 3]), polars.Series([1, 2, 3, 4]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            polars.Series([1, 2, 3]),
            polars.Series([1.0, 2.0, 3.0]),
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
        assert not handler.handle(polars.Series([1, 2, 3]), polars.Series([1, 2, 4]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=polars.Series([1, 2, 3]),
            expected=polars.Series([1, 2, 3, 4]),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.Series have different elements:")


@polars_available
def test_polars_series_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PolarsSeriesEqualHandler().handle(
        polars.Series([0.0, float("nan"), float("nan"), 1.2]),
        polars.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@polars_available
def test_polars_series_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PolarsSeriesEqualHandler().handle(
        polars.Series([0.0, float("nan"), float("nan"), 1.2]),
        polars.Series([0.0, float("nan"), float("nan"), 1.2]),
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


@pytest.mark.parametrize(
    "df_or_series",
    [
        polars.Series(["A", "B", "C"]),
        polars.Series([1, 2, 3]),
        polars.Series([1, 2, 3], dtype=polars.Int64),
        polars.Series([1, 2, 3], dtype=polars.Float64),
        polars.Series([True]),
        polars.DataFrame({"col": [1, 2, 3]}),
        polars.DataFrame({"col": [1, 2, 3]}, schema={"col": polars.Int64}),
        polars.DataFrame({"col": [1, 2, 3]}, schema={"col": polars.Float64}),
        polars.DataFrame({"col": ["A", "B", "C"]}, schema={"col": polars.String}),
    ],
)
def test_has_nan_true(df_or_series: polars.DataFrame | polars.Series) -> None:
    assert not has_nan(df_or_series)


@pytest.mark.parametrize(
    "df_or_series",
    [
        polars.Series([1.0, 2.0, float("nan")]),
        polars.Series([1.0, float("nan"), 3.0], dtype=polars.Float64),
        polars.DataFrame({"col": [1.0, 2.0, float("nan")]}),
        polars.DataFrame({"col": [1.0, 2.0, float("nan")]}, schema={"col": polars.Float64}),
    ],
)
def test_has_nan_false(df_or_series: polars.DataFrame | polars.Series) -> None:
    assert has_nan(df_or_series)

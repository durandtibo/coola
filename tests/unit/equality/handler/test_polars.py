from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    PolarsDataFrameEqualHandler,
    PolarsSeriesEqualHandler,
    TrueHandler,
)
from coola.equality.handler.polars import (
    PolarsLazyFrameEqualHandler,
    frame_equal,
    has_nan,
    series_equal,
)
from coola.testing.fixtures import polars_available
from coola.utils.imports import is_polars_available
from tests.unit.equality.tester.test_polars import (
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
    return EqualityConfig()


#################################################
#     Tests for PolarsDataFrameEqualHandler     #
#################################################


def test_polars_dataframe_equal_handler_repr() -> None:
    assert repr(PolarsDataFrameEqualHandler()) == "PolarsDataFrameEqualHandler()"


def test_polars_dataframe_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(PolarsDataFrameEqualHandler(FalseHandler()))
        == "PolarsDataFrameEqualHandler(next_handler=FalseHandler())"
    )


def test_polars_dataframe_equal_handler_str() -> None:
    assert str(PolarsDataFrameEqualHandler()) == "PolarsDataFrameEqualHandler()"


def test_polars_dataframe_equal_handler_str_with_next_handler() -> None:
    assert str(PolarsDataFrameEqualHandler(FalseHandler())) == "PolarsDataFrameEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PolarsDataFrameEqualHandler(), PolarsDataFrameEqualHandler(), id="without next handler"
        ),
        pytest.param(
            PolarsDataFrameEqualHandler(FalseHandler()),
            PolarsDataFrameEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_polars_dataframe_equal_handler_equal_true(
    handler1: PolarsDataFrameEqualHandler, handler2: PolarsDataFrameEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PolarsDataFrameEqualHandler(TrueHandler()),
            PolarsDataFrameEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            PolarsDataFrameEqualHandler(),
            PolarsDataFrameEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            PolarsDataFrameEqualHandler(FalseHandler()),
            PolarsDataFrameEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(PolarsDataFrameEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_polars_dataframe_equal_handler_equal_false(
    handler1: PolarsDataFrameEqualHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_polars_dataframe_equal_handler_equal_false_different_type_child() -> None:
    class Child(PolarsDataFrameEqualHandler): ...

    assert not PolarsDataFrameEqualHandler().equal(Child())


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
) -> None:
    handler = PolarsDataFrameEqualHandler()
    assert handler.handle(actual, expected, config)


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
def test_polars_dataframe_equal_handler_handle_true_does_not_log(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(actual, expected, config)
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = PolarsDataFrameEqualHandler()
    assert handler.handle(
        pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 3]}), config
    )


@polars_available
def test_polars_dataframe_equal_handler_handle_true_show_difference_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false(
    config: EqualityConfig,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    assert not handler.handle(pl.DataFrame({}), pl.DataFrame({"col": [1, 2, 3]}), config)


@polars_available
def test_polars_dataframe_equal_handler_handle_false_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.DataFrame({}), pl.DataFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_column(
    config: EqualityConfig,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    assert not handler.handle(
        pl.DataFrame({"col1": [1, 2, 3]}), pl.DataFrame({"col2": [1, 2, 3]}), config
    )


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_column_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.DataFrame({"col1": [1, 2, 3]}), pl.DataFrame({"col2": [1, 2, 3]}), config)
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_value(
    config: EqualityConfig,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    assert not handler.handle(
        pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 4]}), config
    )


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_value_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 4]}), config)
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    assert not handler.handle(
        pl.DataFrame(data={"col": [1, 2, 3]}),
        pl.DataFrame(data={"col": [1.0, 2.0, 3.0]}),
        config,
    )


@polars_available
def test_polars_dataframe_equal_handler_handle_false_different_dtype_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pl.DataFrame(data={"col": [1, 2, 3]}),
            pl.DataFrame(data={"col": [1.0, 2.0, 3.0]}),
            config,
        )
        assert not caplog.messages


@polars_available
def test_polars_dataframe_equal_handler_handle_false_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = PolarsDataFrameEqualHandler()
    assert not handler.handle(
        pl.DataFrame({"col": [1, 2, 3]}),
        pl.DataFrame({"col": [1, 2, 4]}),
        config=config,
    )


@polars_available
def test_polars_dataframe_equal_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PolarsDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pl.DataFrame({"col": [1, 2, 3]}),
            pl.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.DataFrames are different:")


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


def test_polars_dataframe_equal_handle_set_next_handler() -> None:
    handler = PolarsDataFrameEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_polars_dataframe_equal_handle_set_next_handler_none() -> None:
    handler = PolarsDataFrameEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_polars_dataframe_equal_handle_set_next_handler_incorrect() -> None:
    handler = PolarsDataFrameEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


#################################################
#     Tests for PolarsLazyFrameEqualHandler     #
#################################################


def test_polars_lazyframe_equal_handler_repr() -> None:
    assert repr(PolarsLazyFrameEqualHandler()) == "PolarsLazyFrameEqualHandler()"


def test_polars_lazyframe_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(PolarsLazyFrameEqualHandler(FalseHandler()))
        == "PolarsLazyFrameEqualHandler(next_handler=FalseHandler())"
    )


def test_polars_lazyframe_equal_handler_str() -> None:
    assert str(PolarsLazyFrameEqualHandler()) == "PolarsLazyFrameEqualHandler()"


def test_polars_lazyframe_equal_handler_str_with_next_handler() -> None:
    assert str(PolarsLazyFrameEqualHandler(FalseHandler())) == "PolarsLazyFrameEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PolarsLazyFrameEqualHandler(), PolarsLazyFrameEqualHandler(), id="without next handler"
        ),
        pytest.param(
            PolarsLazyFrameEqualHandler(FalseHandler()),
            PolarsLazyFrameEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_polars_lazyframe_equal_handler_equal_true(
    handler1: PolarsLazyFrameEqualHandler, handler2: PolarsLazyFrameEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PolarsLazyFrameEqualHandler(TrueHandler()),
            PolarsLazyFrameEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            PolarsLazyFrameEqualHandler(),
            PolarsLazyFrameEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            PolarsLazyFrameEqualHandler(FalseHandler()),
            PolarsLazyFrameEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(PolarsLazyFrameEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_polars_lazyframe_equal_handler_equal_false(
    handler1: PolarsLazyFrameEqualHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_polars_lazyframe_equal_handler_equal_false_different_type_child() -> None:
    class Child(PolarsLazyFrameEqualHandler): ...

    assert not PolarsLazyFrameEqualHandler().equal(Child())


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
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    assert handler.handle(actual, expected, config)


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
def test_polars_lazyframe_equal_handler_handle_true_does_not_log(
    actual: pl.LazyFrame,
    expected: pl.LazyFrame,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(actual, expected, config)
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = PolarsLazyFrameEqualHandler()
    assert handler.handle(
        pl.LazyFrame({"col": [1, 2, 3]}), pl.LazyFrame({"col": [1, 2, 3]}), config
    )


@polars_available
def test_polars_lazyframe_equal_handler_handle_true_show_difference_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.LazyFrame({"col": [1, 2, 3]}), pl.LazyFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false(
    config: EqualityConfig,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    assert not handler.handle(pl.LazyFrame({}), pl.LazyFrame({"col": [1, 2, 3]}), config)


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.LazyFrame({}), pl.LazyFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_column(
    config: EqualityConfig,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    assert not handler.handle(
        pl.LazyFrame({"col1": [1, 2, 3]}), pl.LazyFrame({"col2": [1, 2, 3]}), config
    )


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_column_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.LazyFrame({"col1": [1, 2, 3]}), pl.LazyFrame({"col2": [1, 2, 3]}), config)
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_value(
    config: EqualityConfig,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    assert not handler.handle(
        pl.LazyFrame({"col": [1, 2, 3]}), pl.LazyFrame({"col": [1, 2, 4]}), config
    )


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_value_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.LazyFrame({"col": [1, 2, 3]}), pl.LazyFrame({"col": [1, 2, 4]}), config)
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    assert not handler.handle(
        pl.LazyFrame(data={"col": [1, 2, 3]}),
        pl.LazyFrame(data={"col": [1.0, 2.0, 3.0]}),
        config,
    )


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_different_dtype_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pl.LazyFrame(data={"col": [1, 2, 3]}),
            pl.LazyFrame(data={"col": [1.0, 2.0, 3.0]}),
            config,
        )
        assert not caplog.messages


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = PolarsLazyFrameEqualHandler()
    assert not handler.handle(
        pl.LazyFrame({"col": [1, 2, 3]}),
        pl.LazyFrame({"col": [1, 2, 4]}),
        config=config,
    )


@polars_available
def test_polars_lazyframe_equal_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PolarsLazyFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pl.LazyFrame({"col": [1, 2, 3]}),
            pl.LazyFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.LazyFrames are different:")


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


def test_polars_lazyframe_equal_handle_set_next_handler() -> None:
    handler = PolarsLazyFrameEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_polars_lazyframe_equal_handle_set_next_handler_none() -> None:
    handler = PolarsLazyFrameEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_polars_lazyframe_equal_handle_set_next_handler_incorrect() -> None:
    handler = PolarsLazyFrameEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


##############################################
#     Tests for PolarsSeriesEqualHandler     #
##############################################


def test_polars_series_equal_handler_repr() -> None:
    assert repr(PolarsSeriesEqualHandler()) == "PolarsSeriesEqualHandler()"


def test_polars_series_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(PolarsSeriesEqualHandler(FalseHandler()))
        == "PolarsSeriesEqualHandler(next_handler=FalseHandler())"
    )


def test_polars_series_equal_handler_str() -> None:
    assert str(PolarsSeriesEqualHandler()) == "PolarsSeriesEqualHandler()"


def test_polars_series_equal_handler_str_with_next_handler() -> None:
    assert str(PolarsSeriesEqualHandler(FalseHandler())) == "PolarsSeriesEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PolarsSeriesEqualHandler(), PolarsSeriesEqualHandler(), id="without next handler"
        ),
        pytest.param(
            PolarsSeriesEqualHandler(FalseHandler()),
            PolarsSeriesEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_polars_series_equal_handler_equal_true(
    handler1: PolarsSeriesEqualHandler, handler2: PolarsSeriesEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PolarsSeriesEqualHandler(TrueHandler()),
            PolarsSeriesEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            PolarsSeriesEqualHandler(),
            PolarsSeriesEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            PolarsSeriesEqualHandler(FalseHandler()),
            PolarsSeriesEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(PolarsSeriesEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_polars_series_equal_handler_equal_false(
    handler1: PolarsSeriesEqualHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_polars_series_equal_handler_equal_false_different_type_child() -> None:
    class Child(PolarsSeriesEqualHandler): ...

    assert not PolarsSeriesEqualHandler().equal(Child())


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
) -> None:
    handler = PolarsSeriesEqualHandler()
    assert handler.handle(actual, expected, config)


@polars_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pl.Series([]), pl.Series([])),
        (pl.Series([1, 2, 3]), pl.Series([1, 2, 3])),
        (pl.Series(["a", "b", "c"]), pl.Series(["a", "b", "c"])),
    ],
)
def test_polars_series_equal_handler_handle_true_does_not_log(
    actual: pl.Series,
    expected: pl.Series,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(actual, expected, config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = PolarsSeriesEqualHandler()
    assert handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config)


@polars_available
def test_polars_series_equal_handler_handle_true_show_difference_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_different_shape(
    config: EqualityConfig,
) -> None:
    handler = PolarsSeriesEqualHandler()
    assert not handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 3, 4]), config)


@polars_available
def test_polars_series_equal_handler_handle_false_different_shape_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 3, 4]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
) -> None:
    handler = PolarsSeriesEqualHandler()
    assert not handler.handle(
        pl.Series([1, 2, 3]),
        pl.Series([1.0, 2.0, 3.0]),
        config,
    )


@polars_available
def test_polars_series_equal_handler_handle_false_different_dtype_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pl.Series([1, 2, 3]),
            pl.Series([1.0, 2.0, 3.0]),
            config,
        )
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_different_value(
    config: EqualityConfig,
) -> None:
    handler = PolarsSeriesEqualHandler()
    assert not handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), config)


@polars_available
def test_polars_series_equal_handler_handle_false_different_value_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pl.Series([1, 2, 3]), pl.Series([1, 2, 4]), config)
        assert not caplog.messages


@polars_available
def test_polars_series_equal_handler_handle_false_show_difference(config: EqualityConfig) -> None:
    config.show_difference = True
    handler = PolarsSeriesEqualHandler()
    assert not handler.handle(
        actual=pl.Series([1, 2, 3]),
        expected=pl.Series([1, 2, 3, 4]),
        config=config,
    )


@polars_available
def test_polars_series_equal_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PolarsSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            actual=pl.Series([1, 2, 3]),
            expected=pl.Series([1, 2, 3, 4]),
            config=config,
        )
        assert caplog.messages[0].startswith("polars.Series are different:")


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


def test_polars_series_equal_handle_set_next_handler() -> None:
    handler = PolarsSeriesEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_polars_series_equal_handle_set_next_handler_none() -> None:
    handler = PolarsSeriesEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_polars_series_equal_handle_set_next_handler_incorrect() -> None:
    handler = PolarsSeriesEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


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
    pl.Series([1, 2], dtype=pl.Decimal(10, 2)),
    pl.DataFrame({"col": [1, 2]}, schema={"col": pl.Decimal(10, 2)}),
    pl.DataFrame(
        {"col": [1, 2], "other": [1.0, 2.0]}, schema={"col": pl.Decimal(10, 2), "other": pl.Float64}
    ),
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


@polars_available
def test_polars_frame_equal_different_shape() -> None:
    config = EqualityConfig(equal_nan=True)
    assert not frame_equal(pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [1, 2, 3]}), config)


@polars_available
def test_polars_series_equal_different_shape() -> None:
    config = EqualityConfig(equal_nan=True)
    assert not series_equal(pl.Series([1, 2]), pl.Series([1, 2, 3]), config)


def _make_lazyframes_with_different_schemas() -> tuple[Mock, Mock]:
    actual = Mock(spec=pl.LazyFrame)
    actual.collect_schema.return_value = pl.Schema({"a": pl.Int64})
    expected = Mock(spec=pl.LazyFrame)
    expected.collect_schema.return_value = pl.Schema({"a": pl.Float64})
    return actual, expected


@polars_available
def test_polars_lazyframe_equal_handler_different_schema() -> None:
    actual, expected = _make_lazyframes_with_different_schemas()
    assert not PolarsLazyFrameEqualHandler().handle(actual, expected, EqualityConfig())


@polars_available
def test_polars_lazyframe_equal_handler_different_schema_does_not_collect_actual() -> None:
    actual, expected = _make_lazyframes_with_different_schemas()
    PolarsLazyFrameEqualHandler().handle(actual, expected, EqualityConfig())
    actual.collect.assert_not_called()


@polars_available
def test_polars_lazyframe_equal_handler_different_schema_does_not_collect_expected() -> None:
    actual, expected = _make_lazyframes_with_different_schemas()
    PolarsLazyFrameEqualHandler().handle(actual, expected, EqualityConfig())
    expected.collect.assert_not_called()


@polars_available
def test_has_nan_decimal_with_float_nan() -> None:
    df = pl.DataFrame(
        {"dec": [1, 2], "num": [1.0, float("nan")]},
        schema={"dec": pl.Decimal(10, 2), "num": pl.Float64},
    )
    assert has_nan(df)


@polars_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_frame_equal_decimal(equal_nan: bool) -> None:
    df = pl.DataFrame({"col": [1, 2]}, schema={"col": pl.Decimal(10, 2)})
    assert frame_equal(df, df.clone(), EqualityConfig(equal_nan=equal_nan))


@polars_available
@pytest.mark.parametrize("equal_nan", [True, False])
def test_series_equal_decimal(equal_nan: bool) -> None:
    series = pl.Series([1, 2], dtype=pl.Decimal(10, 2))
    assert series_equal(series, series.clone(), EqualityConfig(equal_nan=equal_nan))


NESTED_NAN_SERIES = [
    pytest.param(pl.Series([[1.0, float("nan")]]), id="list"),
    pytest.param(pl.Series([[1.0, float("nan")]], dtype=pl.Array(pl.Float64, 2)), id="array"),
    pytest.param(pl.Series([{"a": float("nan"), "b": 1}]), id="struct"),
    pytest.param(pl.Series([{"a": [float("nan")]}]), id="struct of list"),
    pytest.param(pl.Series([[[float("nan")]]]), id="list of list"),
]


@polars_available
@pytest.mark.parametrize("series", NESTED_NAN_SERIES)
def test_has_nan_nested_is_left_to_polars(series: pl.Series) -> None:
    assert not has_nan(series)


@polars_available
@pytest.mark.parametrize("series", NESTED_NAN_SERIES)
def test_has_nan_nested_is_left_to_polars_dataframe(series: pl.Series) -> None:
    assert not has_nan(pl.DataFrame({"x": series}))


NESTED_NO_NAN_SERIES = [
    pytest.param(pl.Series([[1.0, 2.0]]), id="list"),
    pytest.param(pl.Series([[], []], dtype=pl.List(pl.Float64)), id="empty lists"),
    pytest.param(pl.Series([None, [1.0]]), id="null list"),
    pytest.param(pl.Series([{"a": 1.0, "b": 1}]), id="struct"),
    pytest.param(pl.Series([[1, 2]]), id="list of int"),
]


@polars_available
@pytest.mark.parametrize("series", NESTED_NO_NAN_SERIES)
def test_has_nan_nested_false(series: pl.Series) -> None:
    assert not has_nan(series)


@polars_available
@pytest.mark.parametrize("series", NESTED_NO_NAN_SERIES)
def test_has_nan_nested_false_dataframe(series: pl.Series) -> None:
    assert not has_nan(pl.DataFrame({"x": series}))


@polars_available
def test_series_equal_nested_nan_follows_polars_without_equal_nan() -> None:
    series = pl.Series([[1.0, float("nan")]])
    assert series_equal(series, series.clone(), EqualityConfig(equal_nan=False))


@polars_available
def test_series_equal_nested_nan_equal_nan() -> None:
    series = pl.Series([[1.0, float("nan")]])
    assert series_equal(series, series.clone(), EqualityConfig(equal_nan=True))


@polars_available
def test_series_equal_nested_floats_have_no_tolerance() -> None:
    assert not series_equal(
        pl.Series([[1.0]]), pl.Series([[1.0000001]]), EqualityConfig(atol=1.0, rtol=1.0)
    )


#############################################
#     Tests for the tolerance of numbers    #
#############################################


@polars_available
def test_series_equal_integers_rtol() -> None:
    assert series_equal(pl.Series([100]), pl.Series([101]), EqualityConfig(rtol=0.01))


@polars_available
def test_series_equal_integers_atol() -> None:
    assert series_equal(pl.Series([1]), pl.Series([2]), EqualityConfig(atol=2.0))


@polars_available
def test_series_equal_integers_outside_atol() -> None:
    assert not series_equal(pl.Series([1]), pl.Series([5]), EqualityConfig(atol=2.0))


@polars_available
def test_series_equal_atol_and_rtol_add_up() -> None:
    # |0 - 0.9| = 0.9 <= 0.5 + 0.5 * 0.9
    assert series_equal(pl.Series([0.0]), pl.Series([0.9]), EqualityConfig(atol=0.5, rtol=0.5))


@polars_available
def test_series_equal_unsigned_integers_do_not_wrap_around() -> None:
    assert not series_equal(
        pl.Series([0], dtype=pl.UInt8), pl.Series([200], dtype=pl.UInt8), EqualityConfig(atol=1.0)
    )


@polars_available
def test_series_equal_numeric_different_names() -> None:
    assert not series_equal(pl.Series("a", [1]), pl.Series("b", [1]), EqualityConfig(atol=1.0))


@polars_available
def test_series_equal_numeric_different_dtypes() -> None:
    assert not series_equal(pl.Series([1]), pl.Series([1.0]), EqualityConfig(atol=1.0))


@polars_available
def test_series_equal_numeric_nulls_at_the_same_positions() -> None:
    assert series_equal(pl.Series([1, None]), pl.Series([2, None]), EqualityConfig(atol=1.0))


@polars_available
def test_series_equal_numeric_nulls_at_different_positions() -> None:
    assert not series_equal(pl.Series([1, None]), pl.Series([None, 1]), EqualityConfig(atol=1.0))


@polars_available
def test_series_equal_numeric_null_and_value() -> None:
    assert not series_equal(pl.Series([1, None]), pl.Series([1, 1]), EqualityConfig(atol=1.0))


@polars_available
def test_series_equal_non_numeric_stays_exact() -> None:
    assert not series_equal(pl.Series(["a"]), pl.Series(["b"]), EqualityConfig(atol=1.0))


@polars_available
def test_frame_equal_integers_atol() -> None:
    assert frame_equal(
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pl.DataFrame({"a": [1, 3], "b": ["x", "y"]}),
        EqualityConfig(atol=1.0),
    )


@polars_available
def test_frame_equal_integers_outside_atol() -> None:
    assert not frame_equal(
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pl.DataFrame({"a": [1, 5], "b": ["x", "y"]}),
        EqualityConfig(atol=1.0),
    )


@polars_available
def test_frame_equal_non_numeric_column_stays_exact() -> None:
    assert not frame_equal(
        pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pl.DataFrame({"a": [1, 2], "b": ["x", "z"]}),
        EqualityConfig(atol=1.0),
    )


@polars_available
def test_frame_equal_different_column_order_with_tolerance() -> None:
    assert not frame_equal(
        pl.DataFrame({"a": [1], "b": [2]}),
        pl.DataFrame({"b": [2], "a": [1]}),
        EqualityConfig(atol=1.0),
    )


@polars_available
def test_frame_equal_different_column_names_with_tolerance() -> None:
    assert not frame_equal(
        pl.DataFrame({"a": [1]}), pl.DataFrame({"b": [1]}), EqualityConfig(atol=1.0)
    )


@polars_available
def test_frame_equal_only_non_numeric_columns_with_tolerance() -> None:
    assert frame_equal(
        pl.DataFrame({"b": ["x"]}), pl.DataFrame({"b": ["x"]}), EqualityConfig(atol=1.0)
    )


@polars_available
def test_frame_equal_numeric_columns_with_equal_nan() -> None:
    frame = pl.DataFrame({"a": [1.0, float("nan")]})
    assert frame_equal(frame, frame.clone(), EqualityConfig(atol=1.0, equal_nan=True))


@polars_available
def test_series_equal_duration_with_tolerance() -> None:
    assert series_equal(
        pl.Series([timedelta(seconds=1)]),
        pl.Series([timedelta(seconds=1.5)]),
        EqualityConfig(atol=1.0),
    )


@polars_available
def test_series_equal_duration_outside_tolerance() -> None:
    assert not series_equal(
        pl.Series([timedelta(seconds=1)]),
        pl.Series([timedelta(seconds=3)]),
        EqualityConfig(atol=1.0),
    )


@polars_available
def test_series_equal_duration_without_tolerance() -> None:
    assert not series_equal(
        pl.Series([timedelta(seconds=1)]), pl.Series([timedelta(seconds=2)]), EqualityConfig()
    )


@polars_available
def test_series_equal_duration_nulls_at_the_same_positions() -> None:
    series = pl.Series([timedelta(seconds=1), None])
    assert series_equal(series, series.clone(), EqualityConfig())


@polars_available
def test_series_equal_duration_nulls_at_different_positions() -> None:
    assert not series_equal(
        pl.Series([timedelta(seconds=1), None]),
        pl.Series([None, timedelta(seconds=1)]),
        EqualityConfig(atol=5.0),
    )


@polars_available
def test_series_equal_duration_different_time_units() -> None:
    assert not series_equal(
        pl.Series([timedelta(seconds=1)], dtype=pl.Duration("ms")),
        pl.Series([timedelta(seconds=1)], dtype=pl.Duration("ns")),
        EqualityConfig(),
    )


@polars_available
def test_series_equal_instants_have_no_tolerance() -> None:
    assert not series_equal(
        pl.Series([datetime(2020, 1, 1, tzinfo=UTC)]),
        pl.Series([datetime(2020, 1, 2, tzinfo=UTC)]),
        EqualityConfig(atol=1e9),
    )


@polars_available
def test_series_equal_different_dtype_non_numeric() -> None:
    assert not series_equal(
        pl.Series(["a"]), pl.Series(["a"], dtype=pl.Categorical), EqualityConfig()
    )


@polars_available
def test_series_equal_null_equals_null() -> None:
    assert series_equal(pl.Series(["a", None]), pl.Series(["a", None]), EqualityConfig())


@polars_available
def test_series_equal_float_nan_without_equal_nan() -> None:
    assert not series_equal(pl.Series([float("nan")]), pl.Series([float("nan")]), EqualityConfig())


@polars_available
def test_frame_equal_different_schema_order() -> None:
    assert not frame_equal(
        pl.DataFrame({"a": [1], "b": ["x"]}), pl.DataFrame({"b": ["x"], "a": [1]}), EqualityConfig()
    )


@polars_available
def test_frame_equal_different_dtype() -> None:
    assert not frame_equal(pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [1.0]}), EqualityConfig())


@polars_available
def test_frame_equal_stops_at_the_first_different_column() -> None:
    df1 = pl.DataFrame({"a": [1.0], "b": [1.0]})
    df2 = pl.DataFrame({"a": [2.0], "b": [1.0]})
    with patch("coola.equality.handler.polars._values_equal", return_value=False) as mock:
        frame_equal(df1, df2, EqualityConfig())
    assert mock.call_count == 1


@polars_available
def test_frame_equal_float_nan_column_without_equal_nan() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan")]})
    assert not frame_equal(df, df.clone(), EqualityConfig())

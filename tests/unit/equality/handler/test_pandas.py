from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig2
from coola.equality.handler import (
    FalseHandler,
    PandasDataFrameEqualHandler,
    PandasSeriesEqualHandler,
)
from coola.testing.fixtures import pandas_available
from coola.utils.imports import is_pandas_available
from tests.unit.equality.comparators.test_pandas import (
    PANDAS_DATAFRAME_EQUAL_TOLERANCE,
    PANDAS_SERIES_EQUAL_TOLERANCE,
)

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    pd = Mock()

if TYPE_CHECKING:
    from tests.unit.equality.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig2:
    return EqualityConfig2()


#################################################
#     Tests for PandasDataFrameEqualHandler     #
#################################################


def test_pandas_dataframe_equal_handler__eq__true() -> None:
    assert PandasDataFrameEqualHandler() == PandasDataFrameEqualHandler()


def test_pandas_dataframe_equal_handler__eq__false_different_type() -> None:
    assert PandasDataFrameEqualHandler() != FalseHandler()


def test_pandas_dataframe_equal_handler__eq__false_different_type_child() -> None:
    class Child(PandasDataFrameEqualHandler): ...

    assert PandasDataFrameEqualHandler() != Child()


def test_pandas_dataframe_equal_handler_repr() -> None:
    assert repr(PandasDataFrameEqualHandler()).startswith("PandasDataFrameEqualHandler(")


def test_pandas_dataframe_equal_handler_str() -> None:
    assert str(PandasDataFrameEqualHandler()).startswith("PandasDataFrameEqualHandler(")


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pd.DataFrame({}), pd.DataFrame({})),
        (pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 3]})),
        (
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
        ),
    ],
)
def test_pandas_dataframe_equal_handler_handle_true(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(actual, expected, config)
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_true_show_difference(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(
            pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 3]}), config
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(pd.DataFrame({}), pd.DataFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_column(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pd.DataFrame({"col1": [1, 2, 3]}), pd.DataFrame({"col2": [1, 2, 3]}), config
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_value(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 4]}), config
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_dtype(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pd.DataFrame(data={"col": [1, 2, 3]}, dtype=float),
            pd.DataFrame(data={"col": [1, 2, 3]}, dtype=int),
            config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_show_difference(
    config: EqualityConfig2, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pd.DataFrame({"col": [1, 2, 3]}),
            pd.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("pandas.DataFrames have different elements:")


@pandas_available
def test_pandas_dataframe_equal_handler_handle_equal_nan_false(config: EqualityConfig2) -> None:
    assert not PandasDataFrameEqualHandler().handle(
        pd.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        pd.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@pandas_available
def test_pandas_dataframe_equal_handler_handle_equal_nan_true(config: EqualityConfig2) -> None:
    config.equal_nan = True
    assert PandasDataFrameEqualHandler().handle(
        pd.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        pd.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL_TOLERANCE)
def test_pandas_dataframe_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig2
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PandasDataFrameEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_pandas_dataframe_equal_handler_set_next_handler() -> None:
    PandasDataFrameEqualHandler().set_next_handler(FalseHandler())


##############################################
#     Tests for PandasSeriesEqualHandler     #
##############################################


def test_pandas_series_equal_handler__eq__true() -> None:
    assert PandasSeriesEqualHandler() == PandasSeriesEqualHandler()


def test_pandas_series_equal_handler__eq__false_different_type() -> None:
    assert PandasSeriesEqualHandler() != FalseHandler()


def test_pandas_series_equal_handler__eq__false_different_type_child() -> None:
    class Child(PandasSeriesEqualHandler): ...

    assert PandasSeriesEqualHandler() != Child()


def test_pandas_series_equal_handler_repr() -> None:
    assert repr(PandasSeriesEqualHandler()).startswith("PandasSeriesEqualHandler(")


def test_pandas_series_equal_handler_str() -> None:
    assert str(PandasSeriesEqualHandler()).startswith("PandasSeriesEqualHandler(")


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pd.Series(data=[], dtype=object), pd.Series(data=[], dtype=object)),
        (pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3])),
        (pd.Series(data=["a", "b", "c"]), pd.Series(data=["a", "b", "c"])),
    ],
)
def test_pandas_series_equal_handler_handle_true(
    actual: pd.Series,
    expected: pd.Series,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(actual, expected, config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_true_show_difference(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3]), config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_shape(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3, 4]), config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_dtype(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pd.Series(data=[1, 2, 3], dtype=int),
            pd.Series(data=[1, 2, 3], dtype=float),
            config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_value(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 4]), config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_index(
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            pd.Series(data=[1, 2, 3]),
            pd.Series(data=[1, 2, 3], index=pd.Index([2, 3, 4])),
            config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_show_difference(
    config: EqualityConfig2, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=pd.Series(data=[1, 2, 3]),
            expected=pd.Series(data=[1, 2, 3, 4]),
            config=config,
        )
        assert caplog.messages[0].startswith("pandas.Series have different elements:")


@pandas_available
def test_pandas_series_equal_handler_handle_equal_nan_false(config: EqualityConfig2) -> None:
    assert not PandasSeriesEqualHandler().handle(
        pd.Series([0.0, float("nan"), float("nan"), 1.2]),
        pd.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pandas_available
def test_pandas_series_equal_handler_handle_equal_nan_true(config: EqualityConfig2) -> None:
    config.equal_nan = True
    assert PandasSeriesEqualHandler().handle(
        pd.Series([0.0, float("nan"), float("nan"), 1.2]),
        pd.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL_TOLERANCE)
def test_pandas_series_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig2
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PandasSeriesEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_pandas_series_equal_handler_set_next_handler() -> None:
    PandasSeriesEqualHandler().set_next_handler(FalseHandler())

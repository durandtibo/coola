from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, PandasSeriesEqualHandler
from coola.testing import pandas_available
from coola.utils import is_pandas_available

if is_pandas_available():
    import pandas
else:  # pragma: no cover
    pandas = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##############################################
#     Tests for PandasSeriesEqualHandler     #
##############################################


def test_pandas_series_equal_handler_eq_true() -> None:
    assert PandasSeriesEqualHandler() == PandasSeriesEqualHandler()


def test_pandas_series_equal_handler_eq_false() -> None:
    assert PandasSeriesEqualHandler() != FalseHandler()


def test_pandas_series_equal_handler_repr() -> None:
    assert repr(PandasSeriesEqualHandler()).startswith("PandasSeriesEqualHandler(")


def test_pandas_series_equal_handler_str() -> None:
    assert str(PandasSeriesEqualHandler()).startswith("PandasSeriesEqualHandler(")


@pandas_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (pandas.Series(data=[], dtype=object), pandas.Series(data=[], dtype=object)),
        (pandas.Series(data=[1, 2, 3]), pandas.Series(data=[1, 2, 3])),
        (pandas.Series(data=["a", "b", "c"]), pandas.Series(data=["a", "b", "c"])),
    ],
)
def test_pandas_series_equal_handler_handle_true(
    object1: pandas.Series,
    object2: pandas.Series,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(object1, object2, config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert handler.handle(pandas.Series(data=[1, 2, 3]), pandas.Series(data=[1, 2, 3]), config)
        assert not caplog.messages


@pandas_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (pandas.Series(data=[1, 2, 3]), pandas.Series(data=[1, 2, 3, 4])),
        (pandas.Series(data=[1, 2, 3], dtype=int), pandas.Series(data=[1, 2, 3], dtype=float)),
        (pandas.Series(data=[1, 2, 3]), pandas.Series(data=[1, 2, 4])),
    ],
)
def test_pandas_series_equal_handler_handle_false(
    object1: pandas.Series,
    object2: pandas.Series,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(object1, object2, config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            object1=pandas.Series(data=[1, 2, 3]),
            object2=pandas.Series(data=[1, 2, 3, 4]),
            config=config,
        )
        assert caplog.messages[0].startswith("pandas.Series have different elements:")


@pandas_available
def test_pandas_series_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PandasSeriesEqualHandler().handle(
        pandas.Series([0.0, float("nan"), float("nan"), 1.2]),
        pandas.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pandas_available
def test_pandas_series_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PandasSeriesEqualHandler().handle(
        pandas.Series([0.0, float("nan"), float("nan"), 1.2]),
        pandas.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


def test_pandas_series_equal_handler_set_next_handler() -> None:
    PandasSeriesEqualHandler().set_next_handler(FalseHandler())

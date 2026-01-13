from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    PolarsDataFrameEqualityTester,
    PolarsLazyFrameEqualityTester,
    PolarsSeriesEqualityTester,
)
from coola.testing.fixtures import polars_available, polars_not_available
from coola.utils.imports import is_polars_available

if is_polars_available():
    import polars as pl


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


###################################################
#     Tests for PolarsDataFrameEqualityTester     #
###################################################


@polars_available
def test_polars_dataframe_equality_tester_with_polars(config: EqualityConfig) -> None:
    assert PolarsDataFrameEqualityTester().objects_are_equal(
        pl.DataFrame({"col": [1, 2, 3]}), pl.DataFrame({"col": [1, 2, 3]}), config=config
    )


@polars_not_available
def test_polars_dataframe_equality_tester_without_polars() -> None:
    with pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."):
        PolarsDataFrameEqualityTester()


###################################################
#     Tests for PolarsLazyFrameEqualityTester     #
###################################################


@polars_available
def test_polars_lazyframe_equality_tester_with_polars(config: EqualityConfig) -> None:
    assert PolarsLazyFrameEqualityTester().objects_are_equal(
        pl.LazyFrame({"col": [1, 2, 3]}), pl.LazyFrame({"col": [1, 2, 3]}), config=config
    )


@polars_not_available
def test_polars_lazyframe_equality_tester_without_polars() -> None:
    with pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."):
        PolarsLazyFrameEqualityTester()


################################################
#     Tests for PolarsSeriesEqualityTester     #
################################################


@polars_available
def test_polars_series_equality_tester_with_polars(config: EqualityConfig) -> None:
    assert PolarsSeriesEqualityTester().objects_are_equal(
        pl.Series([1, 2, 3]), pl.Series([1, 2, 3]), config=config
    )


@polars_not_available
def test_polars_series_equality_tester_without_polars() -> None:
    with pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."):
        PolarsSeriesEqualityTester()

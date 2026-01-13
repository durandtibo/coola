from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    PandasDataFrameEqualityTester,
    PandasSeriesEqualityTester,
)
from coola.testing.fixtures import pandas_available, pandas_not_available
from coola.utils.imports import is_pandas_available

if is_pandas_available():
    import pandas as pd


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


###################################################
#     Tests for PandasDataFrameEqualityTester     #
###################################################


@pandas_available
def test_pandas_dataframe_equality_tester_with_pandas(config: EqualityConfig) -> None:
    assert PandasDataFrameEqualityTester().objects_are_equal(
        pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 3]}), config=config
    )


@pandas_not_available
def test_pandas_dataframe_equality_tester_without_pandas() -> None:
    with pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."):
        PandasDataFrameEqualityTester()


################################################
#     Tests for PandasSeriesEqualityTester     #
################################################


@pandas_available
def test_pandas_series_equality_tester_with_pandas(config: EqualityConfig) -> None:
    assert PandasSeriesEqualityTester().objects_are_equal(
        pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), config=config
    )


@pandas_not_available
def test_pandas_series_equality_tester_without_pandas() -> None:
    with pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."):
        PandasSeriesEqualityTester()

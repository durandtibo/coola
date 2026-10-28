from __future__ import annotations

import math

import pytest

from coola.rounding.polars import round_polars
from coola.testing.marks import polars_available
from coola.utils.imports import is_polars_available

if is_polars_available():
    import polars as pl
    from polars.testing import assert_frame_equal, assert_series_equal

#####################################
#     Tests for round_polars        #
#####################################


@polars_available
def test_round_polars_series_floor() -> None:
    assert_series_equal(
        round_polars(pl.Series([2.567, -2.567, 0.29]), "floor", 2), pl.Series([2.56, -2.57, 0.29])
    )


@polars_available
def test_round_polars_series_ceil() -> None:
    assert_series_equal(
        round_polars(pl.Series([2.561, -2.561, 0.07]), "ceil", 2), pl.Series([2.57, -2.56, 0.07])
    )


@polars_available
def test_round_polars_series_round() -> None:
    assert_series_equal(
        round_polars(pl.Series([0.5, 1.5, 2.5]), "round", 0), pl.Series([0.0, 2.0, 2.0])
    )


@polars_available
def test_round_polars_series_keeps_name() -> None:
    assert round_polars(pl.Series("v", [1.26]), "floor", 1).name == "v"


@polars_available
def test_round_polars_series_null_kept() -> None:
    assert_series_equal(round_polars(pl.Series([1.26, None]), "floor", 1), pl.Series([1.2, None]))


@polars_available
def test_round_polars_series_special_values_kept() -> None:
    result = round_polars(pl.Series([math.inf, math.nan, 1e308]), "floor", 2).to_list()
    assert result[0] == math.inf
    assert math.isnan(result[1])
    assert result[2] == 1e308


@polars_available
def test_round_polars_series_float32_dtype_kept() -> None:
    assert round_polars(pl.Series([1.26], dtype=pl.Float32), "floor", 1).dtype == pl.Float32


@polars_available
def test_round_polars_series_float32_value_on_the_grid() -> None:
    series = pl.Series([0.29, 0.07], dtype=pl.Float32)
    assert_series_equal(round_polars(series, "floor", 2), series)


@polars_available
def test_round_polars_series_integer_unchanged() -> None:
    assert_series_equal(round_polars(pl.Series([1, 2]), "floor", 1), pl.Series([1, 2]))


@polars_available
def test_round_polars_series_empty() -> None:
    assert len(round_polars(pl.Series([], dtype=pl.Float64), "floor", 1)) == 0


@polars_available
def test_round_polars_series_too_many_decimals_unchanged() -> None:
    assert_series_equal(round_polars(pl.Series([0.1]), "floor", 400), pl.Series([0.1]))


@polars_available
@pytest.mark.parametrize("series", [pl.Series(["a"]), pl.Series([True])])
def test_round_polars_series_unsupported_dtype(series: pl.Series) -> None:
    with pytest.raises(TypeError, match=r"Unsupported dtype"):
        round_polars(series, "round", 0)


@polars_available
def test_round_polars_dataframe() -> None:
    frame = pl.DataFrame({"a": [1.26, None], "b": [1, 2], "c": ["x", "y"], "d": [True, False]})
    expected = pl.DataFrame({"a": [1.2, None], "b": [1, 2], "c": ["x", "y"], "d": [True, False]})
    assert_frame_equal(round_polars(frame, "floor", 1), expected)


@polars_available
def test_round_polars_dataframe_does_not_modify_input() -> None:
    frame = pl.DataFrame({"a": [1.26]})
    round_polars(frame, "floor", 1)
    assert_frame_equal(frame, pl.DataFrame({"a": [1.26]}))


@polars_available
def test_round_polars_dataframe_no_numerical_column() -> None:
    frame = pl.DataFrame({"a": ["x"]})
    assert_frame_equal(round_polars(frame, "floor", 1), frame)


@polars_available
def test_round_polars_unsupported_type() -> None:
    with pytest.raises(TypeError, match=r"Unsupported type"):
        round_polars([1.5], "round", 0)

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from coola.rounding.polars import get_default_rounders, round_polars
from coola.testing.marks import polars_available
from coola.utils.imports import MissingPackageError, is_polars_available

if TYPE_CHECKING:
    from coola.rounding.common import RoundMode

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


#######################################
#     Tests for the exact results     #
#######################################


@polars_available
def test_round_polars_series_round_is_exact() -> None:
    result = round_polars(pl.Series([-4.822, 1.2, 4.8, 12.0]), "round", 1)
    assert result.to_list() == [-4.8, 1.2, 4.8, 12.0]


@polars_available
@pytest.mark.parametrize("mode", ["round", "floor", "ceil"])
def test_round_polars_series_second_row_not_affected_by_reciprocal(mode: RoundMode) -> None:
    # polars divides by a scalar as a multiplication by its reciprocal: 12 / 10 is 1.2000000000000002
    assert round_polars(pl.Series([-4.822, 12.0]), mode, 1).to_list()[1] == 12.0


@polars_available
def test_round_polars_series_float32_kept() -> None:
    assert round_polars(pl.Series([1.26], dtype=pl.Float32), "floor", 1).dtype == pl.Float32


@polars_available
def test_round_polars_series_null_and_nan_kept() -> None:
    result = round_polars(pl.Series([1.26, None, math.nan]), "floor", 1)
    assert result.is_null().to_list() == [False, True, False]


@polars_available
def test_round_polars_series_nan_stays_nan() -> None:
    assert round_polars(pl.Series([1.26, None, math.nan]), "floor", 1).is_nan().to_list() == [
        False,
        None,
        True,
    ]


@polars_available
def test_round_polars_series_integer_returned_equal() -> None:
    assert round_polars(pl.Series([1, 2]), "floor", 1).to_list() == [1, 2]


@polars_available
def test_round_polars_dataframe_exact_output() -> None:
    frame = pl.DataFrame({"a": [2.675, 0.125], "b": [1, 2], "c": ["x", "y"], "d": [True, False]})
    assert round_polars(frame, "round", 2).to_dict(as_series=False) == {
        "a": [2.68, 0.12],
        "b": [1, 2],
        "c": ["x", "y"],
        "d": [True, False],
    }


@polars_available
def test_round_polars_requires_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("coola.rounding.polars.check_numpy", _raise_missing_numpy)
    with pytest.raises(MissingPackageError, match=r"numpy"):
        round_polars(pl.Series([1.26]), "floor", 1)


def _raise_missing_numpy() -> None:
    msg = "'numpy' package is required but not installed."
    raise MissingPackageError(msg)


#############################################
#     Tests for get_default_rounders        #
#############################################


@polars_available
def test_get_default_rounders_polars() -> None:
    assert get_default_rounders() == {pl.Series: round_polars, pl.DataFrame: round_polars}


@polars_available
def test_get_default_rounders_polars_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("coola.rounding.polars.is_polars_available", lambda: False)
    assert get_default_rounders() == {}

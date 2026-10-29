from __future__ import annotations

import pytest

from coola.rounding.pandas import get_default_rounders, round_pandas
from coola.testing.marks import pandas_available
from coola.utils.imports import is_pandas_available

if is_pandas_available():
    import pandas as pd
    from pandas.testing import assert_frame_equal, assert_series_equal

#####################################
#     Tests for round_pandas        #
#####################################


@pandas_available
def test_round_pandas_series_floor() -> None:
    assert_series_equal(
        round_pandas(pd.Series([2.567, -2.567, 0.29]), "floor", 2), pd.Series([2.56, -2.57, 0.29])
    )


@pandas_available
def test_round_pandas_series_ceil() -> None:
    assert_series_equal(
        round_pandas(pd.Series([2.561, -2.561, 0.07]), "ceil", 2), pd.Series([2.57, -2.56, 0.07])
    )


@pandas_available
def test_round_pandas_series_round() -> None:
    assert_series_equal(
        round_pandas(pd.Series([0.5, 1.5, 2.5]), "round", 0), pd.Series([0.0, 2.0, 2.0])
    )


@pandas_available
def test_round_pandas_series_keeps_index_and_name() -> None:
    series = pd.Series([1.26, 2.34], index=["x", "y"], name="v")
    assert_series_equal(
        round_pandas(series, "floor", 1), pd.Series([1.2, 2.3], index=["x", "y"], name="v")
    )


@pandas_available
def test_round_pandas_series_nan_kept() -> None:
    assert_series_equal(
        round_pandas(pd.Series([1.26, float("nan")]), "floor", 1), pd.Series([1.2, float("nan")])
    )


@pandas_available
def test_round_pandas_series_nullable_float_na_kept() -> None:
    assert_series_equal(
        round_pandas(pd.Series([1.26, None], dtype="Float64"), "floor", 1),
        pd.Series([1.2, None], dtype="Float64"),
    )


@pandas_available
def test_round_pandas_series_float32_dtype_kept() -> None:
    assert round_pandas(pd.Series([1.26], dtype="float32"), "floor", 1).dtype == "float32"


@pandas_available
def test_round_pandas_series_integer_unchanged() -> None:
    assert_series_equal(round_pandas(pd.Series([1, 2]), "floor", 1), pd.Series([1, 2]))


@pandas_available
def test_round_pandas_series_nullable_integer_unchanged() -> None:
    assert_series_equal(
        round_pandas(pd.Series([1, None], dtype="Int64"), "floor", 1),
        pd.Series([1, None], dtype="Int64"),
    )


@pandas_available
def test_round_pandas_series_does_not_modify_input() -> None:
    series = pd.Series([1.26, 2.5])
    round_pandas(series, "floor", 1)
    assert_series_equal(series, pd.Series([1.26, 2.5]))


@pandas_available
def test_round_pandas_series_empty() -> None:
    assert len(round_pandas(pd.Series([], dtype=float), "floor", 1)) == 0


@pandas_available
@pytest.mark.parametrize(
    "series", [pd.Series(["a"]), pd.Series([True]), pd.Series(pd.to_datetime(["2020-01-01"]))]
)
def test_round_pandas_series_unsupported_dtype(series: pd.Series) -> None:
    with pytest.raises(TypeError, match=r"Unsupported dtype"):
        round_pandas(series, "round", 0)


@pandas_available
def test_round_pandas_dataframe() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.26, float("nan")],
            "b": [1, 2],
            "c": ["x", "y"],
            "d": [True, False],
            "e": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )
    expected = frame.copy()
    expected["a"] = [1.2, float("nan")]
    assert_frame_equal(round_pandas(frame, "floor", 1), expected)


@pandas_available
def test_round_pandas_dataframe_does_not_modify_input() -> None:
    frame = pd.DataFrame({"a": [1.26]})
    round_pandas(frame, "floor", 1)
    assert_frame_equal(frame, pd.DataFrame({"a": [1.26]}))


@pandas_available
def test_round_pandas_dataframe_duplicate_column_names() -> None:
    frame = pd.DataFrame([[1.26, 2.34]], columns=["a", "a"])
    assert_frame_equal(
        round_pandas(frame, "floor", 1), pd.DataFrame([[1.2, 2.3]], columns=["a", "a"])
    )


@pandas_available
def test_round_pandas_dataframe_no_numerical_column() -> None:
    frame = pd.DataFrame({"a": ["x"]})
    assert_frame_equal(round_pandas(frame, "floor", 1), frame)


@pandas_available
def test_round_pandas_unsupported_type() -> None:
    with pytest.raises(TypeError, match=r"Unsupported type"):
        round_pandas([1.5], "round", 0)


#############################################
#     Tests for get_default_rounders        #
#############################################


@pandas_available
def test_get_default_rounders_pandas() -> None:
    assert get_default_rounders() == {pd.Series: round_pandas, pd.DataFrame: round_pandas}


@pandas_available
def test_get_default_rounders_pandas_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("coola.rounding.pandas.is_pandas_available", lambda: False)
    assert get_default_rounders() == {}

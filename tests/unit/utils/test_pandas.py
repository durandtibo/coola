import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark

from coola import AllCloseTester, EqualityTester
from coola._pandas import (
    DataFrameEqualityOperator,
    SeriesAllCloseOperator,
    SeriesEqualityOperator,
)
from coola.testing import pandas_available
from coola.utils.imports import is_pandas_available

if is_pandas_available():
    import pandas as pd
else:
    pd = Mock()

###############################################
#     Tests for DataFrameEqualityOperator     #
###############################################


@pandas_available
def test_dataframe_equality_operator_str() -> None:
    assert str(DataFrameEqualityOperator()).startswith("DataFrameEqualityOperator(")


@pandas_available
def test_dataframe_equality_operator_equal_true() -> None:
    assert DataFrameEqualityOperator().equal(
        EqualityTester(),
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
    )


@pandas_available
def test_dataframe_equality_operator_equal_true_same_object() -> None:
    obj = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
        }
    )
    assert DataFrameEqualityOperator().equal(EqualityTester(), obj, obj)


@pandas_available
def test_dataframe_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert DataFrameEqualityOperator().equal(
            EqualityTester(),
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
            show_difference=True,
        )
        assert not caplog.messages


@pandas_available
def test_dataframe_equality_operator_equal_false_different_data() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
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
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_different_dtype() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        pd.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_nan() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, float("nan")],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, float("nan")],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameEqualityOperator().equal(
            EqualityTester(),
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
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("pandas.DataFrames are different")


@pandas_available
def test_dataframe_equality_operator_equal_false_different_type() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
        "meow",
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameEqualityOperator().equal(
            EqualityTester(),
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            "meow",
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a pandas.DataFrame")


############################################
#     Tests for SeriesAllCloseOperator     #
############################################


@pandas_available
def test_series_allclose_operator_str() -> None:
    assert str(SeriesAllCloseOperator()).startswith("SeriesAllCloseOperator(")


@pandas_available
def test_series_allclose_operator_equal_true() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), pd.Series([1, 2, 3, 4, 5]), pd.Series([1, 2, 3, 4, 5])
    )


@pandas_available
def test_series_allclose_operator_allclose_true_same_object() -> None:
    obj = pd.Series([1, 2, 3, 4, 5])
    assert SeriesAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@pandas_available
def test_series_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert SeriesAllCloseOperator().allclose(
            AllCloseTester(),
            pd.Series([1, 2, 3, 4, 5]),
            pd.Series([1, 2, 3, 4, 5]),
            show_difference=True,
        )
        assert not caplog.messages


@pandas_available
def test_series_allclose_operator_allclose_false_different_data() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), pd.Series([1, 2, 3, 4, 5]), pd.Series(["a", "b", "c", "d", "e"])
    )


@pandas_available
def test_series_allclose_operator_allclose_false_different_dtype() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), pd.Series([1, 2, 3, 4, 5]), pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    )


@pandas_available
def test_series_allclose_operator_allclose_false_nan() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pd.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        pd.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
    )


@pandas_available
def test_series_allclose_operator_allclose_true_nan() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pd.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        pd.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        equal_nan=True,
    )


@pandas_available
def test_series_allclose_operator_allclose_false_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesAllCloseOperator().allclose(
            AllCloseTester(),
            pd.Series([1, 2, 3, 4, 5]),
            pd.Series(["a", "b", "c", "d", "e"]),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("pandas.Series are different")


@pandas_available
def test_series_allclose_operator_allclose_false_different_type() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), pd.Series([1, 2, 3, 4, 5]), "meow"
    )


@pandas_available
def test_series_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesAllCloseOperator().allclose(
            AllCloseTester(), pd.Series([1, 2, 3, 4, 5]), "meow", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a pandas.Series")


@pandas_available
@mark.parametrize(
    "series,atol",
    (
        (pd.Series([1.5, 1.5, 1.5]), 1.0),
        (pd.Series([1.05, 1.05, 1.05]), 1e-1),
        (pd.Series([1.005, 1.005, 1.005]), 1e-2),
    ),
)
def test_ndarray_allclose_operator_allclose_true_atol(series: pd.Series, atol: float) -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), pd.Series([1.0, 1.0, 1.0]), series, atol=atol, rtol=0
    )


@pandas_available
@mark.parametrize(
    "series,rtol",
    (
        (pd.Series([1.5, 1.5, 1.5]), 1.0),
        (pd.Series([1.05, 1.05, 1.05]), 1e-1),
        (pd.Series([1.005, 1.005, 1.005]), 1e-2),
    ),
)
def test_ndarray_allclose_operator_allclose_true_rtol(series: pd.Series, rtol: float) -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), pd.Series([1.0, 1.0, 1.0]), series, rtol=rtol
    )


############################################
#     Tests for SeriesEqualityOperator     #
############################################


@pandas_available
def test_series_equality_operator_str() -> None:
    assert str(SeriesEqualityOperator()).startswith("SeriesEqualityOperator(")


@pandas_available
def test_series_equality_operator_equal_true() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(), pd.Series([1, 2, 3, 4, 5]), pd.Series([1, 2, 3, 4, 5])
    )


@pandas_available
def test_series_equality_operator_equal_true_same_object() -> None:
    obj = pd.Series([1, 2, 3, 4, 5])
    assert SeriesEqualityOperator().equal(EqualityTester(), obj, obj)


@pandas_available
def test_series_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert SeriesEqualityOperator().equal(
            EqualityTester(),
            pd.Series([1, 2, 3, 4, 5]),
            pd.Series([1, 2, 3, 4, 5]),
            show_difference=True,
        )
        assert not caplog.messages


@pandas_available
def test_series_equality_operator_equal_false_different_data() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(), pd.Series([1, 2, 3, 4, 5]), pd.Series(["a", "b", "c", "d", "e"])
    )


@pandas_available
def test_series_equality_operator_equal_false_different_dtype() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(), pd.Series([1, 2, 3, 4, 5]), pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    )


@pandas_available
def test_series_equality_operator_equal_false_nan() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(),
        pd.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        pd.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
    )


@pandas_available
def test_series_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesEqualityOperator().equal(
            EqualityTester(),
            pd.Series([1, 2, 3, 4, 5]),
            pd.Series(["a", "b", "c", "d", "e"]),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("pandas.Series are different")


@pandas_available
def test_series_equality_operator_equal_false_different_type() -> None:
    assert not SeriesEqualityOperator().equal(EqualityTester(), pd.Series([1, 2, 3, 4, 5]), "meow")


@pandas_available
def test_series_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesEqualityOperator().equal(
            EqualityTester(), pd.Series([1, 2, 3, 4, 5]), "meow", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a pandas.Series")

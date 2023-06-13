import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture

from coola import EqualityTester, objects_are_equal
from coola._polars import SeriesEqualityOperator
from coola.testing import polars_available
from coola.utils.imports import is_polars_available

if is_polars_available():
    import polars
else:
    polars = Mock()


@polars_available
def test_equality_tester_registry() -> None:
    # assert isinstance(EqualityTester.registry[polars.DataFrame], DataFrameEqualityOperator)
    assert isinstance(EqualityTester.registry[polars.Series], SeriesEqualityOperator)


############################################
#     Tests for SeriesEqualityOperator     #
############################################


@polars_available
def test_objects_are_equal_series() -> None:
    assert objects_are_equal(polars.Series([1, 2, 3, 4, 5]), polars.Series([1, 2, 3, 4, 5]))


@polars_available
def test_series_equality_operator_str() -> None:
    assert str(SeriesEqualityOperator()).startswith("SeriesEqualityOperator(")


@polars_available
def test_series_equality_operator_equal_true_int() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(), polars.Series([1, 2, 3, 4, 5]), polars.Series([1, 2, 3, 4, 5])
    )


@polars_available
def test_series_equality_operator_equal_true_float() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(),
        polars.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        polars.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


@polars_available
def test_series_equality_operator_equal_true_str() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(),
        polars.Series(["a", "b", "c", "d", "e"]),
        polars.Series(["a", "b", "c", "d", "e"]),
    )


@polars_available
def test_series_equality_operator_equal_true_datetime() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14"]).str.to_datetime(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14"]).str.to_datetime(),
    )


@polars_available
def test_series_equality_operator_equal_true_same_object() -> None:
    obj = polars.Series([1, 2, 3, 4, 5])
    assert SeriesEqualityOperator().equal(EqualityTester(), obj, obj)


@polars_available
def test_series_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert SeriesEqualityOperator().equal(
            EqualityTester(),
            polars.Series([1, 2, 3, 4, 5]),
            polars.Series([1, 2, 3, 4, 5]),
            show_difference=True,
        )
        assert not caplog.messages


@polars_available
def test_series_equality_operator_equal_false_different_data() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(), polars.Series([1, 2, 3, 4, 5]), polars.Series(["a", "b", "c", "d", "e"])
    )


@polars_available
def test_series_equality_operator_equal_false_different_dtype() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(), polars.Series([1, 2, 3, 4, 5]), polars.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    )


@polars_available
def test_series_equality_operator_equal_false_nan() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
    )


@polars_available
def test_series_equality_operator_equal_false_nat() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
    )


@polars_available
def test_series_equality_operator_equal_false_none() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(),
        polars.Series(["a", "b", "c", "d", "e", None]),
        polars.Series(["a", "b", "c", "d", "e", None]),
    )


@polars_available
def test_series_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesEqualityOperator().equal(
            EqualityTester(),
            polars.Series([1, 2, 3, 4, 5]),
            polars.Series(["a", "b", "c", "d", "e"]),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("polars.Series are different")


@polars_available
def test_series_equality_operator_equal_false_different_type() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(), polars.Series([1, 2, 3, 4, 5]), "meow"
    )


@polars_available
def test_series_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesEqualityOperator().equal(
            EqualityTester(), polars.Series([1, 2, 3, 4, 5]), "meow", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a polars.Series")

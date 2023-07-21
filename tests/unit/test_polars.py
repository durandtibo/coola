import logging
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark, raises

from coola import (
    AllCloseTester,
    EqualityTester,
    objects_are_allclose,
    objects_are_equal,
)
from coola.polars_ import (
    DataFrameAllCloseOperator,
    DataFrameEqualityOperator,
    SeriesAllCloseOperator,
    SeriesEqualityOperator,
)
from coola.testing import polars_available
from coola.utils.imports import is_polars_available

if is_polars_available():
    import polars
else:
    polars = Mock()


@polars_available
def test_allclose_tester_registry() -> None:
    assert isinstance(AllCloseTester.registry[polars.DataFrame], DataFrameAllCloseOperator)
    assert isinstance(AllCloseTester.registry[polars.Series], SeriesAllCloseOperator)


@polars_available
def test_equality_tester_registry() -> None:
    assert isinstance(EqualityTester.registry[polars.DataFrame], DataFrameEqualityOperator)
    assert isinstance(EqualityTester.registry[polars.Series], SeriesEqualityOperator)


###############################################
#     Tests for DataFrameAllCloseOperator     #
###############################################


@polars_available
def test_objects_are_allclose_dataframe() -> None:
    assert objects_are_allclose(
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_allclose_operator_str() -> None:
    assert str(DataFrameAllCloseOperator()).startswith("DataFrameAllCloseOperator(")


@polars_available
def test_dataframe_allclose_operator__eq__true() -> None:
    assert DataFrameAllCloseOperator() == DataFrameAllCloseOperator()


@polars_available
def test_dataframe_allclose_operator__eq__false() -> None:
    assert DataFrameAllCloseOperator() != 123


@polars_available
def test_dataframe_allclose_operator_allclose_true() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_true_same_object() -> None:
    obj = polars.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": polars.Series(
                ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
            ).str.to_datetime(),
        }
    )
    assert DataFrameAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@polars_available
def test_dataframe_allclose_operator_allclose_true_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert DataFrameAllCloseOperator().allclose(
            AllCloseTester(),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            show_difference=True,
        )
        assert not caplog.messages


@polars_available
def test_dataframe_allclose_operator_allclose_false_different_data() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_different_columns() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_different_dtype() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_null() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_true_null() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
        equal_nan=True,
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_nan() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
        polars.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_nat() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {"col": polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime()}
        ),
        polars.DataFrame(
            {"col": polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime()}
        ),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_true_nat() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {"col": polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime()}
        ),
        polars.DataFrame(
            {"col": polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime()}
        ),
        equal_nan=True,
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_none_str() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_true_none_str() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        equal_nan=True,
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_none_int() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
    )


@polars_available
def test_dataframe_allclose_operator_allclose_true_none_int() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        equal_nan=True,
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameAllCloseOperator().allclose(
            AllCloseTester(),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("polars.DataFrames are different")


@polars_available
def test_dataframe_allclose_operator_allclose_false_different_type() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        "meow",
    )


@polars_available
def test_dataframe_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameAllCloseOperator().allclose(
            AllCloseTester(),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            "meow",
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("object2 is not a polars.DataFrame")


@polars_available
@mark.parametrize(
    "df,atol",
    (
        (
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.5, 2.5, 3.5, 4.5, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            1.0,
        ),
        (
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.05, 2.05, 3.05, 4.05, 5.05],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            1e-1,
        ),
        (
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.005, 2.005, 3.005, 4.005, 5.005],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            1e-2,
        ),
    ),
)
def test_dataframe_allclose_operator_allclose_true_atol(df: polars.DataFrame, atol: float) -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        df,
        atol=atol,
        rtol=0.0,
    )


@polars_available
@mark.parametrize(
    "df,rtol",
    (
        (
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.5, 2.5, 3.5, 4.5, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            1.0,
        ),
        (
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.05, 2.15, 3.25, 4.35, 5.45],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            1e-1,
        ),
        (
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.005, 2.015, 3.025, 4.035, 5.045],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            1e-2,
        ),
    ),
)
def test_dataframe_allclose_operator_allclose_true_rtol(df: polars.DataFrame, rtol: float) -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        df,
        atol=0.0,
        rtol=rtol,
    )


@polars_available
def test_dataframe_allclose_operator_clone() -> None:
    op = DataFrameAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@polars_available
def test_dataframe_allclose_operator_no_polars() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`polars` package is required but not installed."):
            DataFrameAllCloseOperator()


###############################################
#     Tests for DataFrameEqualityOperator     #
###############################################


@polars_available
def test_objects_are_equal_dataframe() -> None:
    assert objects_are_equal(
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_str() -> None:
    assert str(DataFrameEqualityOperator()).startswith("DataFrameEqualityOperator(")


@polars_available
def test_dataframe_equality_operator__eq__true() -> None:
    assert DataFrameEqualityOperator() == DataFrameEqualityOperator()


@polars_available
def test_dataframe_equality_operator__eq__false_different_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True) != DataFrameEqualityOperator(
        nulls_compare_equal=False
    )


@polars_available
def test_dataframe_equality_operator__eq__false_different_type() -> None:
    assert DataFrameEqualityOperator() != 123


@polars_available
@mark.parametrize("nulls_compare_equal", (True, False))
def test_dataframe_equality_operator_clone(nulls_compare_equal: bool) -> None:
    op = DataFrameEqualityOperator(nulls_compare_equal)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@polars_available
def test_dataframe_equality_operator_equal_true() -> None:
    assert DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_true_same_object() -> None:
    obj = polars.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": polars.Series(
                ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
            ).str.to_datetime(),
        }
    )
    assert DataFrameEqualityOperator().equal(EqualityTester(), obj, obj)


@polars_available
def test_dataframe_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert DataFrameEqualityOperator().equal(
            EqualityTester(),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            show_difference=True,
        )
        assert not caplog.messages


@polars_available
def test_dataframe_equality_operator_equal_false_different_data() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_different_columns() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_different_dtype() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_null() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_nan() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
        polars.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_nat() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame(
            {"col": polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime()}
        ),
        polars.DataFrame(
            {"col": polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime()}
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_none_str() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_none_int() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameEqualityOperator().equal(
            EqualityTester(),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("polars.DataFrames are different")


@polars_available
def test_dataframe_equality_operator_equal_false_different_type() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime(),
            }
        ),
        "meow",
    )


@polars_available
def test_dataframe_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameEqualityOperator().equal(
            EqualityTester(),
            polars.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": polars.Series(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ).str.to_datetime(),
                }
            ),
            "meow",
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a polars.DataFrame")


@polars_available
def test_dataframe_equality_operator_equal_true_null_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_null_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 6, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
        polars.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ).str.to_datetime(),
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_true_nan_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
        polars.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_nan_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
        polars.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 6.0, float("nan")]}),
    )


@polars_available
def test_dataframe_equality_operator_equal_true_nat_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime()
            }
        ),
        polars.DataFrame(
            {
                "col": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime()
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_nat_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame(
            {
                "col": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ).str.to_datetime()
            }
        ),
        polars.DataFrame(
            {
                "col": polars.Series(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/18"]
                ).str.to_datetime()
            }
        ),
    )


@polars_available
def test_dataframe_equality_operator_equal_true_none_str_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_none_str_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        polars.DataFrame({"col": ["a", "b", "c", "d", "f", None]}),
    )


@polars_available
def test_dataframe_equality_operator_equal_true_none_int_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
    )


@polars_available
def test_dataframe_equality_operator_equal_false_none_int_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        polars.DataFrame({"col": [1, 2, 3, 4, 6, None]}),
    )


@polars_available
def test_dataframe_equality_operator_no_polars() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`polars` package is required but not installed."):
            DataFrameEqualityOperator()


############################################
#     Tests for SeriesAllCloseOperator     #
############################################


@polars_available
def test_objects_are_allclose_series() -> None:
    assert objects_are_allclose(polars.Series([1, 2, 3, 4, 5]), polars.Series([1, 2, 3, 4, 5]))


@polars_available
def test_series_allclose_operator_str() -> None:
    assert str(SeriesAllCloseOperator()).startswith("SeriesAllCloseOperator(")


@polars_available
def test_series_allclose_operator__eq__true() -> None:
    assert SeriesAllCloseOperator() == SeriesAllCloseOperator()


@polars_available
def test_series_allclose_operator__eq__false() -> None:
    assert SeriesAllCloseOperator() != 123


@polars_available
def test_series_allclose_operator_allclose_true_int() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), polars.Series([1, 2, 3, 4, 5]), polars.Series([1, 2, 3, 4, 5])
    )


@polars_available
def test_series_allclose_operator_allclose_true_float() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        polars.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


@polars_available
def test_series_allclose_operator_allclose_true_str() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series(["a", "b", "c", "d", "e"]),
        polars.Series(["a", "b", "c", "d", "e"]),
    )


@polars_available
def test_series_allclose_operator_allclose_true_datetime() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14"]).str.to_datetime(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14"]).str.to_datetime(),
    )


@polars_available
def test_series_allclose_operator_allclose_true_same_object() -> None:
    obj = polars.Series([1, 2, 3, 4, 5])
    assert SeriesAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@polars_available
def test_series_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert SeriesAllCloseOperator().allclose(
            AllCloseTester(),
            polars.Series([1, 2, 3, 4, 5]),
            polars.Series([1, 2, 3, 4, 5]),
            show_difference=True,
        )
        assert not caplog.messages


@polars_available
def test_series_allclose_operator_allclose_false_different_data() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), polars.Series([1, 2, 3, 4, 5]), polars.Series(["a", "b", "c", "d", "e"])
    )


@polars_available
def test_series_allclose_operator_allclose_false_different_dtype() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), polars.Series([1, 2, 3, 4, 5]), polars.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    )


@polars_available
def test_series_allclose_operator_allclose_false_nan() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
    )


@polars_available
def test_series_allclose_operator_allclose_true_nan() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        equal_nan=True,
    )


@polars_available
def test_series_allclose_operator_allclose_false_nat() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
    )


@polars_available
def test_series_allclose_operator_allclose_true_nat() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
        equal_nan=True,
    )


@polars_available
def test_series_allclose_operator_allclose_false_none() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series(["a", "b", "c", "d", "e", None]),
        polars.Series(["a", "b", "c", "d", "e", None]),
    )


@polars_available
def test_series_allclose_operator_allclose_true_none() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        polars.Series(["a", "b", "c", "d", "e", None]),
        polars.Series(["a", "b", "c", "d", "e", None]),
        equal_nan=True,
    )


@polars_available
def test_series_allclose_operator_allclose_false_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesAllCloseOperator().allclose(
            AllCloseTester(),
            polars.Series([1, 2, 3, 4, 5]),
            polars.Series(["a", "b", "c", "d", "e"]),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("polars.Series are different")


@polars_available
def test_series_allclose_operator_allclose_false_different_type() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), polars.Series([1, 2, 3, 4, 5]), "meow"
    )


@polars_available
def test_series_allclose_operator_allclose_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesAllCloseOperator().allclose(
            AllCloseTester(), polars.Series([1, 2, 3, 4, 5]), "meow", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a polars.Series")


@polars_available
@mark.parametrize(
    "series,atol",
    (
        (polars.Series([1.5, 1.5, 1.5]), 1.0),
        (polars.Series([1.05, 1.05, 1.05]), 1e-1),
        (polars.Series([1.005, 1.005, 1.005]), 1e-2),
    ),
)
def test_series_allclose_operator_allclose_true_atol(series: polars.Series, atol: float) -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), polars.Series([1.0, 1.0, 1.0]), series, atol=atol, rtol=0.0
    )


@polars_available
@mark.parametrize(
    "series,rtol",
    (
        (polars.Series([1.5, 1.5, 1.5]), 1.0),
        (polars.Series([1.05, 1.05, 1.05]), 1e-1),
        (polars.Series([1.005, 1.005, 1.005]), 1e-2),
    ),
)
def test_series_allclose_operator_allclose_true_rtol(series: polars.Series, rtol: float) -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), polars.Series([1.0, 1.0, 1.0]), series, atol=0.0, rtol=rtol
    )


@polars_available
def test_series_allclose_operator_clone() -> None:
    op = SeriesAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@polars_available
def test_series_allclose_operator_no_polars() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`polars` package is required but not installed."):
            SeriesAllCloseOperator()


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
def test_series_equality_operator__eq__true() -> None:
    assert SeriesEqualityOperator() == SeriesEqualityOperator()


@polars_available
def test_series_equality_operator__eq__false_different_nulls_compare_equal() -> None:
    assert SeriesEqualityOperator(nulls_compare_equal=True) != SeriesEqualityOperator(
        nulls_compare_equal=False
    )


@polars_available
def test_series_equality_operator__eq__false_different_type() -> None:
    assert SeriesEqualityOperator() != 123


@polars_available
@mark.parametrize("nulls_compare_equal", (True, False))
def test_series_equality_operator_clone(nulls_compare_equal: bool) -> None:
    op = SeriesEqualityOperator(nulls_compare_equal)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


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


@polars_available
def test_series_equality_operator_equal_true_nan_nulls_compare_equal() -> None:
    assert SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
    )


@polars_available
def test_series_equality_operator_equal_false_nan_nulls_compare_equal() -> None:
    assert not SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        polars.Series([1.0, 2.0, 3.0, 5.0, float("nan")]),
    )


@polars_available
def test_series_equality_operator_equal_true_nat_nulls_compare_equal() -> None:
    assert SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
    )


@polars_available
def test_series_equality_operator_equal_false_nat_nulls_compare_equal() -> None:
    assert not SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/14", None]).str.to_datetime(),
        polars.Series(["2020/10/12", "2021/3/14", "2022/4/16", None]).str.to_datetime(),
    )


@polars_available
def test_series_equality_operator_equal_true_none_nulls_compare_equal() -> None:
    assert SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.Series(["a", "b", "c", "d", "e", None]),
        polars.Series(["a", "b", "c", "d", "e", None]),
    )


@polars_available
def test_series_equality_operator_equal_false_none_nulls_compare_equal() -> None:
    assert not SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        polars.Series(["a", "b", "c", "d", "e", None]),
        polars.Series(["a", "b", "c", "d", "f", None]),
    )


@polars_available
def test_series_equality_operator_no_polars() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`polars` package is required but not installed."):
            SeriesEqualityOperator()

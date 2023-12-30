import logging
from unittest.mock import Mock, patch

import pytest

from coola import (
    AllCloseTester,
    EqualityTester,
    objects_are_allclose,
    objects_are_equal,
)
from coola.comparators.pandas_ import (
    DataFrameAllCloseOperator,
    DataFrameEqualityOperator,
    SeriesAllCloseOperator,
    SeriesEqualityOperator,
    get_mapping_allclose,
    get_mapping_equality,
)
from coola.testing import pandas_available
from coola.utils.imports import is_pandas_available

if is_pandas_available():
    import pandas
else:
    pandas = Mock()


@pandas_available
def test_allclose_tester_registry() -> None:
    assert isinstance(AllCloseTester.registry[pandas.DataFrame], DataFrameAllCloseOperator)
    assert isinstance(AllCloseTester.registry[pandas.Series], SeriesAllCloseOperator)


@pandas_available
def test_equality_tester_registry() -> None:
    assert isinstance(EqualityTester.registry[pandas.DataFrame], DataFrameEqualityOperator)
    assert isinstance(EqualityTester.registry[pandas.Series], SeriesEqualityOperator)


###############################################
#     Tests for DataFrameAllCloseOperator     #
###############################################


@pandas_available
def test_objects_are_allclose_dataframe() -> None:
    assert objects_are_allclose(
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_allclose_operator_str() -> None:
    assert str(DataFrameAllCloseOperator()).startswith("DataFrameAllCloseOperator(")


@pandas_available
def test_dataframe_allclose_operator__eq__true() -> None:
    assert DataFrameAllCloseOperator() == DataFrameAllCloseOperator()


@pandas_available
def test_dataframe_allclose_operator__eq__false() -> None:
    assert DataFrameAllCloseOperator() != 123


@pandas_available
def test_dataframe_allclose_operator_allclose_true() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_true_same_object() -> None:
    obj = pandas.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": pandas.to_datetime(
                ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
            ),
        }
    )
    assert DataFrameAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@pandas_available
def test_dataframe_allclose_operator_allclose_true_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert DataFrameAllCloseOperator().allclose(
            AllCloseTester(),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            show_difference=True,
        )
        assert not caplog.messages


@pandas_available
def test_dataframe_allclose_operator_allclose_false_different_data() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_different_columns() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_different_index() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            },
            index=pandas.Index([2, 3, 4, 5, 6]),
        ),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_different_dtype() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_null() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_true_null() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
        equal_nan=True,
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_nan() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
        pandas.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_nat() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {"col": pandas.to_datetime(["2020/10/12", "2021/3/14", "2022/4/14", None])}
        ),
        pandas.DataFrame(
            {"col": pandas.to_datetime(["2020/10/12", "2021/3/14", "2022/4/14", None])}
        ),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_true_nat() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {"col": pandas.to_datetime(["2020/10/12", "2021/3/14", "2022/4/14", None])}
        ),
        pandas.DataFrame(
            {"col": pandas.to_datetime(["2020/10/12", "2021/3/14", "2022/4/14", None])}
        ),
        equal_nan=True,
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_none_str() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_true_none_str() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        equal_nan=True,
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_none_int() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_true_none_int() -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        equal_nan=True,
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameAllCloseOperator().allclose(
            AllCloseTester(),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("pandas.DataFrames are different")


@pandas_available
def test_dataframe_allclose_operator_allclose_false_different_type() -> None:
    assert not DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        "meow",
    )


@pandas_available
def test_dataframe_allclose_operator_allclose_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameAllCloseOperator().allclose(
            AllCloseTester(),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            "meow",
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("object2 is not a pandas.DataFrame")


@pandas_available
@pytest.mark.parametrize(
    ("df", "atol"),
    [
        (
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.5, 2.5, 3.5, 4.5, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            1.0,
        ),
        (
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.05, 2.05, 3.05, 4.05, 5.05],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            1e-1,
        ),
        (
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.005, 2.005, 3.005, 4.005, 5.005],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            1e-2,
        ),
    ],
)
def test_dataframe_allclose_operator_allclose_true_atol(df: pandas.DataFrame, atol: float) -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        df,
        atol=atol,
        rtol=0.0,
    )


@pandas_available
@pytest.mark.parametrize(
    ("df", "rtol"),
    [
        (
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.5, 2.5, 3.5, 4.5, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            1.0,
        ),
        (
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.05, 2.15, 3.25, 4.35, 5.45],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            1e-1,
        ),
        (
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.005, 2.015, 3.025, 4.035, 5.045],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            1e-2,
        ),
    ],
)
def test_dataframe_allclose_operator_allclose_true_rtol(df: pandas.DataFrame, rtol: float) -> None:
    assert DataFrameAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        df,
        atol=0.0,
        rtol=rtol,
    )


@pandas_available
def test_dataframe_allclose_operator_clone() -> None:
    op = DataFrameAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@pandas_available
def test_dataframe_allclose_operator_no_pandas() -> None:
    with patch(
        "coola.utils.imports.is_pandas_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`pandas` package is required but not installed."):
        DataFrameAllCloseOperator()


###############################################
#     Tests for DataFrameEqualityOperator     #
###############################################


@pandas_available
def test_objects_are_equal_dataframe() -> None:
    assert objects_are_equal(
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_str() -> None:
    assert str(DataFrameEqualityOperator()).startswith("DataFrameEqualityOperator(")


@pandas_available
def test_dataframe_equality_operator__eq__true() -> None:
    assert DataFrameEqualityOperator() == DataFrameEqualityOperator()


@pandas_available
def test_dataframe_equality_operator__eq__false_different_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True) != DataFrameEqualityOperator(
        nulls_compare_equal=False
    )


@pandas_available
def test_dataframe_equality_operator__eq__false_different_type() -> None:
    assert DataFrameEqualityOperator() != 123


@pandas_available
@pytest.mark.parametrize("nulls_compare_equal", [True, False])
def test_dataframe_equality_operator_clone(nulls_compare_equal: bool) -> None:
    op = DataFrameEqualityOperator(nulls_compare_equal)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@pandas_available
def test_dataframe_equality_operator_equal_true() -> None:
    assert DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_true_same_object() -> None:
    obj = pandas.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
            "col3": ["a", "b", "c", "d", "e"],
            "col4": pandas.to_datetime(
                ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
            ),
        }
    )
    assert DataFrameEqualityOperator().equal(EqualityTester(), obj, obj)


@pandas_available
def test_dataframe_equality_operator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert DataFrameEqualityOperator().equal(
            EqualityTester(),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            show_difference=True,
        )
        assert not caplog.messages


@pandas_available
def test_dataframe_equality_operator_equal_false_different_data() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_different_columns() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_different_index() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            },
            index=pandas.Index([2, 3, 4, 5, 6]),
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_different_dtype() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_null() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_nan() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
        pandas.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_nat() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                )
            }
        ),
        pandas.DataFrame(
            {
                "col": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                )
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_none_str() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_none_int() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameEqualityOperator().equal(
            EqualityTester(),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("pandas.DataFrames are different")


@pandas_available
def test_dataframe_equality_operator_equal_false_different_type() -> None:
    assert not DataFrameEqualityOperator().equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "col3": ["a", "b", "c", "d", "e"],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                ),
            }
        ),
        "meow",
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not DataFrameEqualityOperator().equal(
            EqualityTester(),
            pandas.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                    "col4": pandas.to_datetime(
                        ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16"]
                    ),
                }
            ),
            "meow",
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a pandas.DataFrame")


@pandas_available
def test_dataframe_equality_operator_equal_true_null_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_null_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 6, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
        pandas.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, None],
                "col2": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")],
                "col3": ["a", "b", "c", "d", "e", None],
                "col4": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                ),
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_true_nan_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
        pandas.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_nan_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 5.5, float("nan")]}),
        pandas.DataFrame({"col": [1.1, 2.2, 3.3, 4.4, 6.0, float("nan")]}),
    )


@pandas_available
def test_dataframe_equality_operator_equal_true_nat_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                )
            }
        ),
        pandas.DataFrame(
            {
                "col": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                )
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_nat_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame(
            {
                "col": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/16", None]
                )
            }
        ),
        pandas.DataFrame(
            {
                "col": pandas.to_datetime(
                    ["2020/10/12", "2021/3/14", "2022/4/14", "2023/5/15", "2024/6/17", None]
                )
            }
        ),
    )


@pandas_available
def test_dataframe_equality_operator_equal_true_none_str_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_none_str_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "e", None]}),
        pandas.DataFrame({"col": ["a", "b", "c", "d", "f", None]}),
    )


@pandas_available
def test_dataframe_equality_operator_equal_true_none_int_nulls_compare_equal() -> None:
    assert DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
    )


@pandas_available
def test_dataframe_equality_operator_equal_false_none_int_nulls_compare_equal() -> None:
    assert not DataFrameEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.DataFrame({"col": [1, 2, 3, 4, 5, None]}),
        pandas.DataFrame({"col": [1, 2, 3, 4, 6, None]}),
    )


@pandas_available
def test_dataframe_equality_operator_no_pandas() -> None:
    with patch(
        "coola.utils.imports.is_pandas_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`pandas` package is required but not installed."):
        DataFrameEqualityOperator()


############################################
#     Tests for SeriesAllCloseOperator     #
############################################


@pandas_available
def test_objects_are_allclose_series() -> None:
    assert objects_are_allclose(pandas.Series([1, 2, 3, 4, 5]), pandas.Series([1, 2, 3, 4, 5]))


@pandas_available
def test_series_allclose_operator_str() -> None:
    assert str(SeriesAllCloseOperator()).startswith("SeriesAllCloseOperator(")


@pandas_available
def test_series_allclose_operator__eq__true() -> None:
    assert SeriesAllCloseOperator() == SeriesAllCloseOperator()


@pandas_available
def test_series_allclose_operator__eq__false() -> None:
    assert SeriesAllCloseOperator() != 123


@pandas_available
def test_series_allclose_operator_allclose_true_int() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), pandas.Series([1, 2, 3, 4, 5]), pandas.Series([1, 2, 3, 4, 5])
    )


@pandas_available
def test_series_allclose_operator_allclose_true_float() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pandas.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


@pandas_available
def test_series_allclose_operator_allclose_true_str() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.Series(["a", "b", "c", "d", "e"]),
        pandas.Series(["a", "b", "c", "d", "e"]),
    )


@pandas_available
def test_series_allclose_operator_allclose_true_datetime() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14"])),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14"])),
    )


@pandas_available
def test_series_allclose_operator_allclose_true_same_object() -> None:
    obj = pandas.Series([1, 2, 3, 4, 5])
    assert SeriesAllCloseOperator().allclose(AllCloseTester(), obj, obj)


@pandas_available
def test_series_allclose_operator_allclose_true_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert SeriesAllCloseOperator().allclose(
            AllCloseTester(),
            pandas.Series([1, 2, 3, 4, 5]),
            pandas.Series([1, 2, 3, 4, 5]),
            show_difference=True,
        )
        assert not caplog.messages


@pandas_available
def test_series_allclose_operator_allclose_false_different_data() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), pandas.Series([1, 2, 3, 4, 5]), pandas.Series(["a", "b", "c", "d", "e"])
    )


@pandas_available
def test_series_allclose_operator_allclose_false_different_dtype() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), pandas.Series([1, 2, 3, 4, 5]), pandas.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    )


@pandas_available
def test_series_allclose_operator_allclose_false_different_index() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.Series([1, 2, 3, 4, 5]),
        pandas.Series([1, 2, 3, 4, 5], index=pandas.Index([1, 2, 3, 4, 5])),
    )


@pandas_available
def test_series_allclose_operator_allclose_false_nan() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
    )


@pandas_available
def test_series_allclose_operator_allclose_true_nan() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        equal_nan=True,
    )


@pandas_available
def test_series_allclose_operator_allclose_false_nat() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
    )


@pandas_available
def test_series_allclose_operator_allclose_true_nat() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
        equal_nan=True,
    )


@pandas_available
def test_series_allclose_operator_allclose_false_none() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.Series(["a", "b", "c", "d", "e", None]),
        pandas.Series(["a", "b", "c", "d", "e", None]),
    )


@pandas_available
def test_series_allclose_operator_allclose_true_none() -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(),
        pandas.Series(["a", "b", "c", "d", "e", None]),
        pandas.Series(["a", "b", "c", "d", "e", None]),
        equal_nan=True,
    )


@pandas_available
def test_series_allclose_operator_allclose_false_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesAllCloseOperator().allclose(
            AllCloseTester(),
            pandas.Series([1, 2, 3, 4, 5]),
            pandas.Series(["a", "b", "c", "d", "e"]),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("pandas.Series are different")


@pandas_available
def test_series_allclose_operator_allclose_false_different_type() -> None:
    assert not SeriesAllCloseOperator().allclose(
        AllCloseTester(), pandas.Series([1, 2, 3, 4, 5]), "meow"
    )


@pandas_available
def test_series_allclose_operator_allclose_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesAllCloseOperator().allclose(
            AllCloseTester(), pandas.Series([1, 2, 3, 4, 5]), "meow", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a pandas.Series")


@pandas_available
@pytest.mark.parametrize(
    ("series", "atol"),
    [
        (pandas.Series([1.5, 1.5, 1.5]), 1.0),
        (pandas.Series([1.05, 1.05, 1.05]), 1e-1),
        (pandas.Series([1.005, 1.005, 1.005]), 1e-2),
    ],
)
def test_series_allclose_operator_allclose_true_atol(series: pandas.Series, atol: float) -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), pandas.Series([1.0, 1.0, 1.0]), series, atol=atol, rtol=0
    )


@pandas_available
@pytest.mark.parametrize(
    ("series", "rtol"),
    [
        (pandas.Series([1.5, 1.5, 1.5]), 1.0),
        (pandas.Series([1.05, 1.05, 1.05]), 1e-1),
        (pandas.Series([1.005, 1.005, 1.005]), 1e-2),
    ],
)
def test_series_allclose_operator_allclose_true_rtol(series: pandas.Series, rtol: float) -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), pandas.Series([1.0, 1.0, 1.0]), series, atol=0.0, rtol=rtol
    )


@pandas_available
def test_series_allclose_operator_clone() -> None:
    op = SeriesAllCloseOperator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@pandas_available
def test_series_allclose_operator_no_pandas() -> None:
    with patch(
        "coola.utils.imports.is_pandas_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`pandas` package is required but not installed."):
        SeriesAllCloseOperator()


############################################
#     Tests for SeriesEqualityOperator     #
############################################


@pandas_available
def test_objects_are_equal_series() -> None:
    assert objects_are_equal(pandas.Series([1, 2, 3, 4, 5]), pandas.Series([1, 2, 3, 4, 5]))


@pandas_available
def test_series_equality_operator_str() -> None:
    assert str(SeriesEqualityOperator()).startswith("SeriesEqualityOperator(")


@pandas_available
def test_series_equality_operator__eq__true() -> None:
    assert SeriesEqualityOperator() == SeriesEqualityOperator()


@pandas_available
def test_series_equality_operator__eq__false_different_nulls_compare_equal() -> None:
    assert SeriesEqualityOperator(nulls_compare_equal=True) != SeriesEqualityOperator(
        nulls_compare_equal=False
    )


@pandas_available
def test_series_equality_operator__eq__false_different_type() -> None:
    assert SeriesEqualityOperator() != 123


@pandas_available
@pytest.mark.parametrize("nulls_compare_equal", [True, False])
def test_series_equality_operator_clone(nulls_compare_equal: bool) -> None:
    op = SeriesEqualityOperator(nulls_compare_equal)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@pandas_available
def test_series_equality_operator_equal_true_int() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(), pandas.Series([1, 2, 3, 4, 5]), pandas.Series([1, 2, 3, 4, 5])
    )


@pandas_available
def test_series_equality_operator_equal_true_float() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(),
        pandas.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        pandas.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
    )


@pandas_available
def test_series_equality_operator_equal_true_str() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(),
        pandas.Series(["a", "b", "c", "d", "e"]),
        pandas.Series(["a", "b", "c", "d", "e"]),
    )


@pandas_available
def test_series_equality_operator_equal_true_datetime() -> None:
    assert SeriesEqualityOperator().equal(
        EqualityTester(),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14"])),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14"])),
    )


@pandas_available
def test_series_equality_operator_equal_true_same_object() -> None:
    obj = pandas.Series([1, 2, 3, 4, 5])
    assert SeriesEqualityOperator().equal(EqualityTester(), obj, obj)


@pandas_available
def test_series_equality_operator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert SeriesEqualityOperator().equal(
            EqualityTester(),
            pandas.Series([1, 2, 3, 4, 5]),
            pandas.Series([1, 2, 3, 4, 5]),
            show_difference=True,
        )
        assert not caplog.messages


@pandas_available
def test_series_equality_operator_equal_false_different_data() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(), pandas.Series([1, 2, 3, 4, 5]), pandas.Series(["a", "b", "c", "d", "e"])
    )


@pandas_available
def test_series_equality_operator_equal_false_different_dtype() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(), pandas.Series([1, 2, 3, 4, 5]), pandas.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    )


@pandas_available
def test_series_equality_operator_equal_false_nan() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
    )


@pandas_available
def test_series_equality_operator_equal_false_nat() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
    )


@pandas_available
def test_series_equality_operator_equal_false_none() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(),
        pandas.Series(["a", "b", "c", "d", "e", None]),
        pandas.Series(["a", "b", "c", "d", "e", None]),
    )


@pandas_available
def test_series_equality_operator_equal_false_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesEqualityOperator().equal(
            EqualityTester(),
            pandas.Series([1, 2, 3, 4, 5]),
            pandas.Series(["a", "b", "c", "d", "e"]),
            show_difference=True,
        )
        assert caplog.messages[-1].startswith("pandas.Series are different")


@pandas_available
def test_series_equality_operator_equal_false_different_type() -> None:
    assert not SeriesEqualityOperator().equal(
        EqualityTester(), pandas.Series([1, 2, 3, 4, 5]), "meow"
    )


@pandas_available
def test_series_equality_operator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesEqualityOperator().equal(
            EqualityTester(), pandas.Series([1, 2, 3, 4, 5]), "meow", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a pandas.Series")


@pandas_available
def test_series_equality_operator_equal_false_nulls_compare_equal_nan() -> None:
    assert not SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        pandas.Series([1.0, 2.0, 3.0, 5.0, float("nan")]),
    )


@pandas_available
def test_series_equality_operator_equal_true_nulls_compare_equal_nan() -> None:
    assert SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
        pandas.Series([1.0, 2.0, 3.0, 4.0, float("nan")]),
    )


@pandas_available
def test_series_equality_operator_equal_false_nulls_compare_equal_nat() -> None:
    assert not SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/16", None])),
    )


@pandas_available
def test_series_equality_operator_equal_true_nulls_compare_equal_nat() -> None:
    assert SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
        pandas.to_datetime(pandas.Series(["2020/10/12", "2021/3/14", "2022/4/14", None])),
    )


@pandas_available
def test_series_equality_operator_equal_false_nulls_compare_equal_none() -> None:
    assert not SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.Series(["a", "b", "c", "d", "e", None]),
        pandas.Series(["a", "b", "c", "d", "f", None]),
    )


@pandas_available
def test_series_equality_operator_equal_true_nulls_compare_equal_none() -> None:
    assert SeriesEqualityOperator(nulls_compare_equal=True).equal(
        EqualityTester(),
        pandas.Series(["a", "b", "c", "d", "e", None]),
        pandas.Series(["a", "b", "c", "d", "e", None]),
    )


@pandas_available
def test_series_equality_operator_no_pandas() -> None:
    with patch(
        "coola.utils.imports.is_pandas_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`pandas` package is required but not installed."):
        SeriesEqualityOperator()


##########################################
#     Tests for get_mapping_allclose     #
##########################################


@pandas_available
def test_get_mapping_allclose() -> None:
    mapping = get_mapping_allclose()
    assert len(mapping) == 2
    assert isinstance(mapping[pandas.DataFrame], DataFrameAllCloseOperator)
    assert isinstance(mapping[pandas.Series], SeriesAllCloseOperator)


def test_get_mapping_allclose_no_numpy() -> None:
    with patch("coola.comparators.pandas_.is_pandas_available", lambda *args, **kwargs: False):
        assert get_mapping_allclose() == {}


##########################################
#     Tests for get_mapping_equality     #
##########################################


@pandas_available
def test_get_mapping_equality() -> None:
    mapping = get_mapping_equality()
    assert len(mapping) == 2
    assert isinstance(mapping[pandas.DataFrame], DataFrameEqualityOperator)
    assert isinstance(mapping[pandas.Series], SeriesEqualityOperator)


def test_get_mapping_equality_no_numpy() -> None:
    with patch("coola.comparators.pandas_.is_pandas_available", lambda *args, **kwargs: False):
        assert get_mapping_equality() == {}

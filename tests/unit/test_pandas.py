import logging
from unittest.mock import Mock

from pytest import LogCaptureFixture, mark

from coola import (
    AllCloseTester,
    EqualityTester,
    objects_are_allclose,
    objects_are_equal,
)
from coola._pandas import (
    DataFrameAllCloseOperator,
    DataFrameEqualityOperator,
    SeriesAllCloseOperator,
    SeriesEqualityOperator,
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
    caplog: LogCaptureFixture,
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
def test_dataframe_allclose_operator_allclose_false_nan() -> None:
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
def test_dataframe_allclose_operator_allclose_true_nan() -> None:
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
def test_dataframe_allclose_operator_allclose_false_show_difference(
    caplog: LogCaptureFixture,
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
    caplog: LogCaptureFixture,
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
def test_dataframe_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
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
def test_dataframe_equality_operator_equal_false_nan() -> None:
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
def test_dataframe_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture) -> None:
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
    caplog: LogCaptureFixture,
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
def test_series_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture) -> None:
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
def test_series_allclose_operator_allclose_false_show_difference(caplog: LogCaptureFixture) -> None:
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
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesAllCloseOperator().allclose(
            AllCloseTester(), pandas.Series([1, 2, 3, 4, 5]), "meow", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a pandas.Series")


@pandas_available
@mark.parametrize(
    "series,atol",
    (
        (pandas.Series([1.5, 1.5, 1.5]), 1.0),
        (pandas.Series([1.05, 1.05, 1.05]), 1e-1),
        (pandas.Series([1.005, 1.005, 1.005]), 1e-2),
    ),
)
def test_series_allclose_operator_allclose_true_atol(series: pandas.Series, atol: float) -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), pandas.Series([1.0, 1.0, 1.0]), series, atol=atol, rtol=0
    )


@pandas_available
@mark.parametrize(
    "series,rtol",
    (
        (pandas.Series([1.5, 1.5, 1.5]), 1.0),
        (pandas.Series([1.05, 1.05, 1.05]), 1e-1),
        (pandas.Series([1.005, 1.005, 1.005]), 1e-2),
    ),
)
def test_series_allclose_operator_allclose_true_rtol(series: pandas.Series, rtol: float) -> None:
    assert SeriesAllCloseOperator().allclose(
        AllCloseTester(), pandas.Series([1.0, 1.0, 1.0]), series, rtol=rtol
    )


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
def test_series_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
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
def test_series_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture) -> None:
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
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not SeriesEqualityOperator().equal(
            EqualityTester(), pandas.Series([1, 2, 3, 4, 5]), "meow", show_difference=True
        )
        assert caplog.messages[0].startswith("object2 is not a pandas.Series")

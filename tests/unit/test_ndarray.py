import logging
from unittest.mock import patch

import numpy as np
from pytest import LogCaptureFixture, mark, raises

from coola.allclose import AllCloseTester
from coola.equality import EqualityTester
from coola.ndarray import NDArrayAllCloseOperator, NDArrayEqualityOperator

#############################################
#     Tests for NDArrayAllCloseOperator     #
#############################################


def test_ndarray_allclose_operator_str():
    assert str(NDArrayAllCloseOperator()) == "NDArrayAllCloseOperator()"


@mark.parametrize("array", (np.ones((2, 3)), np.ones((2, 3)) + 1e-9, np.ones((2, 3)) - 1e-9))
def test_ndarray_allclose_operator_allclose_true(array: np.array):
    assert NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array)


def test_ndarray_allclose_operator_equal_true_check_dtype_false():
    assert NDArrayAllCloseOperator(check_dtype=False).allclose(
        AllCloseTester(),
        np.ones((2, 3), dtype=float),
        np.ones((2, 3), dtype=int),
    )


def test_ndarray_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


def test_ndarray_allclose_operator_equal_false_different_dtype():
    assert not NDArrayAllCloseOperator().allclose(
        AllCloseTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


def test_ndarray_allclose_operator_equal_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3), dtype=float),
            object2=np.ones((2, 3), dtype=int),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray data types are different:")


def test_ndarray_allclose_operator_equal_false_different_shape():
    assert not NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), np.zeros((6,)))


def test_ndarray_allclose_operator_equal_false_different_shape_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((6,)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray shapes are different:")


@mark.parametrize("array", (np.zeros((2, 3)), np.ones((2, 3)) + 1e-7, np.ones((2, 3)) - 1e-7))
def test_ndarray_allclose_operator_allclose_false_different_value(array: np.array):
    assert not NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array, rtol=0)


def test_ndarray_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarrays are different")


def test_ndarray_allclose_operator_allclose_false_different_type():
    assert not NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), 42)


def test_ndarray_allclose_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a numpy.ndarray:")


@mark.parametrize(
    "array,atol",
    ((np.ones((2, 3)) + 0.5, 1), (np.ones((2, 3)) + 0.05, 1e-1), (np.ones((2, 3)) + 5e-3, 1e-2)),
)
def test_ndarray_allclose_operator_allclose_true_atol(array: np.ndarray, atol: float):
    assert NDArrayAllCloseOperator().allclose(
        AllCloseTester(), np.ones((2, 3)), array, atol=atol, rtol=0
    )


@mark.parametrize(
    "array,rtol",
    ((np.ones((2, 3)) + 0.5, 1), (np.ones((2, 3)) + 0.05, 1e-1), (np.ones((2, 3)) + 5e-3, 1e-2)),
)
def test_ndarray_allclose_operator_allclose_true_rtol(array: np.ndarray, rtol: float):
    assert NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array, rtol=rtol)


def test_ndarray_allclose_operator_no_numpy():
    with patch("coola.ndarray.is_numpy_available", lambda *args, **kwargs: False):
        with raises(RuntimeError):
            NDArrayAllCloseOperator()


#############################################
#     Tests for NDArrayEqualityOperator     #
#############################################


def test_ndarray_equality_operator_str():
    assert str(NDArrayEqualityOperator()) == "NDArrayEqualityOperator()"


def test_ndarray_equality_operator_equal_true():
    assert NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.ones((2, 3)))


def test_ndarray_equality_operator_equal_true_check_dtype_false():
    assert NDArrayEqualityOperator(check_dtype=False).equal(
        EqualityTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


def test_ndarray_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


def test_ndarray_equality_operator_equal_false_different_dtype():
    assert not NDArrayEqualityOperator().equal(
        EqualityTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


def test_ndarray_equality_operator_equal_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3), dtype=float),
            object2=np.ones((2, 3), dtype=int),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray data types are different:")


def test_ndarray_equality_operator_equal_false_different_shape():
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.zeros((6,)))


def test_ndarray_equality_operator_equal_false_different_shape_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((6,)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray shapes are different:")


def test_ndarray_equality_operator_equal_false_different_value():
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.zeros((2, 3)))


def test_ndarray_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarrays are different")


def test_ndarray_equality_operator_equal_false_different_type():
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), 42)


def test_ndarray_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
):
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a numpy.ndarray:")


def test_ndarray_equality_operator_no_numpy():
    with patch("coola.ndarray.is_numpy_available", lambda *args, **kwargs: False):
        with raises(RuntimeError):
            NDArrayEqualityOperator()

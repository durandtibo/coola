import logging

import numpy as np
from pytest import LogCaptureFixture

from coola.equality import EqualityTester
from coola.ndarray import NDArrayEqualityOperator

#############################################
#     Tests for NDArrayEqualityOperator     #
#############################################


def test_ndarray_equality_operator_str():
    assert str(NDArrayEqualityOperator()) == "NDArrayEqualityOperator()"


def test_ndarray_equality_operator_equal_true():
    assert NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.ones((2, 3)))


def test_ndarray_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


def test_ndarray_equality_operator_equal_false_different_value():
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.zeros((2, 3)))


def test_ndarray_equality_operator_equal_false_different_type():
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), 42)


def test_ndarray_equality_operator_equal_false_show_difference(caplog: LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.arrays are different")

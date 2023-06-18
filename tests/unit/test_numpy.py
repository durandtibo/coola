from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark, raises

from coola import is_numpy_available
from coola._numpy import NDArrayAllCloseOperator, NDArrayEqualityOperator
from coola.allclose import AllCloseTester
from coola.equality import EqualityTester
from coola.testing import numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


#############################################
#     Tests for NDArrayAllCloseOperator     #
#############################################


@numpy_available
def test_ndarray_allclose_operator_str() -> None:
    assert str(NDArrayAllCloseOperator()).startswith("NDArrayAllCloseOperator(")


@numpy_available
def test_ndarray_allclose_operator__eq__true() -> None:
    assert NDArrayAllCloseOperator() == NDArrayAllCloseOperator()


@numpy_available
def test_ndarray_allclose_operator__eq__false_different_check_dtype() -> None:
    assert NDArrayAllCloseOperator(check_dtype=True) != NDArrayAllCloseOperator(check_dtype=False)


@numpy_available
def test_ndarray_allclose_operator__eq__false_different_type() -> None:
    assert NDArrayAllCloseOperator() != 123


@numpy_available
@mark.parametrize(
    "array", (np.ones((2, 3)), np.full((2, 3), 1.0 + 1e-9), np.full((2, 3), 1.0 - 1e-9))
)
def test_ndarray_allclose_operator_allclose_true(array: np.ndarray) -> None:
    assert NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array)


@numpy_available
def test_ndarray_allclose_operator_allclose_true_same_object() -> None:
    array = np.ones((2, 3))
    assert NDArrayAllCloseOperator().allclose(AllCloseTester(), array, array)


@numpy_available
def test_ndarray_allclose_operator_equal_true_check_dtype_false() -> None:
    assert NDArrayAllCloseOperator(check_dtype=False).allclose(
        AllCloseTester(),
        np.ones((2, 3), dtype=float),
        np.ones((2, 3), dtype=int),
    )


@numpy_available
def test_ndarray_allclose_operator_allclose_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


@numpy_available
def test_ndarray_allclose_operator_equal_true_nan_equal_nan_true() -> None:
    assert NDArrayAllCloseOperator().allclose(
        AllCloseTester(),
        np.array([0.0, 1.0, float("nan")]),
        np.array([0.0, 1.0, float("nan")]),
        equal_nan=True,
    )


@numpy_available
def test_ndarray_allclose_operator_equal_false_nan_equal_nan_false() -> None:
    assert not NDArrayAllCloseOperator().allclose(
        AllCloseTester(),
        np.array([0.0, 1.0, float("nan")]),
        np.array([0.0, 1.0, float("nan")]),
    )


@numpy_available
def test_ndarray_allclose_operator_equal_false_different_dtype() -> None:
    assert not NDArrayAllCloseOperator().allclose(
        AllCloseTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


@numpy_available
def test_ndarray_allclose_operator_equal_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3), dtype=float),
            object2=np.ones((2, 3), dtype=int),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray data types are different:")


@numpy_available
def test_ndarray_allclose_operator_equal_false_different_shape() -> None:
    assert not NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), np.zeros((6,)))


@numpy_available
def test_ndarray_allclose_operator_equal_false_different_shape_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((6,)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray shapes are different:")


@numpy_available
@mark.parametrize(
    "array", (np.zeros((2, 3)), np.full((2, 3), 1.0 + 1e-7), np.full((2, 3), 1.0 - 1e-7))
)
def test_ndarray_allclose_operator_allclose_false_different_value(array: np.ndarray) -> None:
    assert not NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array, rtol=0)


@numpy_available
def test_ndarray_allclose_operator_allclose_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarrays are different")


@numpy_available
def test_ndarray_allclose_operator_allclose_false_different_type() -> None:
    assert not NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), 42)


@numpy_available
def test_ndarray_allclose_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NDArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a numpy.ndarray:")


@numpy_available
@mark.parametrize(
    "array,atol",
    ((np.full((2, 3), 1.5), 1), (np.full((2, 3), 1.05), 1e-1), (np.full((2, 3), 1.005), 1e-2)),
)
def test_ndarray_allclose_operator_allclose_true_atol(array: np.ndarray, atol: float) -> None:
    assert NDArrayAllCloseOperator().allclose(
        AllCloseTester(), np.ones((2, 3)), array, atol=atol, rtol=0
    )


@numpy_available
@mark.parametrize(
    "array,rtol",
    ((np.full((2, 3), 1.5), 1), (np.full((2, 3), 1.05), 1e-1), (np.full((2, 3), 1.005), 1e-2)),
)
def test_ndarray_allclose_operator_allclose_true_rtol(array: np.ndarray, rtol: float) -> None:
    assert NDArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array, rtol=rtol)


@numpy_available
def test_ndarray_allclose_operator_no_numpy() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`numpy` package is required but not installed."):
            NDArrayAllCloseOperator()


@numpy_available
@mark.parametrize("check_dtype", (True, False))
def test_ndarray_allclose_operator_clone(check_dtype: bool) -> None:
    op = NDArrayAllCloseOperator(check_dtype)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


#############################################
#     Tests for NDArrayEqualityOperator     #
#############################################


@numpy_available
def test_ndarray_equality_operator_str() -> None:
    assert str(NDArrayEqualityOperator()).startswith("NDArrayEqualityOperator(")


@numpy_available
def test_ndarray_equality_operator__eq__true() -> None:
    assert NDArrayEqualityOperator() == NDArrayEqualityOperator()


@numpy_available
def test_ndarray_equality_operator__eq__false_different_check_dtype() -> None:
    assert NDArrayEqualityOperator(check_dtype=True) != NDArrayEqualityOperator(check_dtype=False)


@numpy_available
def test_ndarray_equality_operator__eq__false_different_type() -> None:
    assert NDArrayEqualityOperator() != 123


@numpy_available
@mark.parametrize("check_dtype", (True, False))
def test_ndarray_equality_operator_clone(check_dtype: bool) -> None:
    op = NDArrayEqualityOperator(check_dtype)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@numpy_available
def test_ndarray_equality_operator_equal_true() -> None:
    assert NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.ones((2, 3)))


@numpy_available
def test_ndarray_equality_operator_equal_true_same_object() -> None:
    array = np.ones((2, 3))
    assert NDArrayEqualityOperator().equal(EqualityTester(), array, array)


@numpy_available
def test_ndarray_equality_operator_equal_true_check_dtype_false() -> None:
    assert NDArrayEqualityOperator(check_dtype=False).equal(
        EqualityTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


@numpy_available
def test_ndarray_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        assert NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


@numpy_available
def test_ndarray_equality_operator_equal_false_different_dtype() -> None:
    assert not NDArrayEqualityOperator().equal(
        EqualityTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


@numpy_available
def test_ndarray_equality_operator_equal_false_different_dtype_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3), dtype=float),
            object2=np.ones((2, 3), dtype=int),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray data types are different:")


@numpy_available
def test_ndarray_equality_operator_equal_false_different_shape() -> None:
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.zeros((6,)))


@numpy_available
def test_ndarray_equality_operator_equal_false_different_shape_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((6,)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray shapes are different:")


@numpy_available
def test_ndarray_equality_operator_equal_false_different_value() -> None:
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.zeros((2, 3)))


@numpy_available
def test_ndarray_equality_operator_equal_false_different_value_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarrays are different")


@numpy_available
def test_ndarray_equality_operator_equal_false_different_type() -> None:
    assert not NDArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), 42)


@numpy_available
def test_ndarray_equality_operator_equal_false_different_type_show_difference(
    caplog: LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not NDArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a numpy.ndarray:")


@numpy_available
def test_ndarray_equality_operator_no_numpy() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`numpy` package is required but not installed."):
            NDArrayEqualityOperator()

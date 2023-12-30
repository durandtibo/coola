from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import objects_are_allclose, objects_are_equal
from coola.comparators import ArrayAllCloseOperator, ArrayEqualityOperator
from coola.comparators.numpy_ import get_mapping_allclose, get_mapping_equality
from coola.testers import AllCloseTester, EqualityTester
from coola.testing import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@numpy_available
def test_allclose_tester_registry() -> None:
    assert isinstance(AllCloseTester.registry[np.ndarray], ArrayAllCloseOperator)


@numpy_available
def test_equality_tester_registry() -> None:
    assert isinstance(EqualityTester.registry[np.ndarray], ArrayEqualityOperator)


###########################################
#     Tests for ArrayAllCloseOperator     #
###########################################


@numpy_available
def test_objects_are_allclose_array() -> None:
    assert objects_are_allclose(np.ones((2, 3)), np.ones((2, 3)))


@numpy_available
def test_array_allclose_operator_str() -> None:
    assert str(ArrayAllCloseOperator()).startswith("ArrayAllCloseOperator(")


@numpy_available
def test_array_allclose_operator__eq__true() -> None:
    assert ArrayAllCloseOperator() == ArrayAllCloseOperator()


@numpy_available
def test_array_allclose_operator__eq__false_different_check_dtype() -> None:
    assert ArrayAllCloseOperator(check_dtype=True) != ArrayAllCloseOperator(check_dtype=False)


@numpy_available
def test_array_allclose_operator__eq__false_different_type() -> None:
    assert ArrayAllCloseOperator() != 123


@numpy_available
@pytest.mark.parametrize(
    "array", [np.ones((2, 3)), np.full((2, 3), 1.0 + 1e-9), np.full((2, 3), 1.0 - 1e-9)]
)
def test_array_allclose_operator_allclose_true(array: np.ndarray) -> None:
    assert ArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array)


@numpy_available
def test_array_allclose_operator_allclose_true_same_object() -> None:
    array = np.ones((2, 3))
    assert ArrayAllCloseOperator().allclose(AllCloseTester(), array, array)


@numpy_available
def test_array_allclose_operator_allclose_true_check_dtype_false() -> None:
    assert ArrayAllCloseOperator(check_dtype=False).allclose(
        AllCloseTester(),
        np.ones((2, 3), dtype=float),
        np.ones((2, 3), dtype=int),
    )


@numpy_available
def test_array_allclose_operator_allclose_true_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert ArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


@numpy_available
def test_array_allclose_operator_allclose_true_nan_equal_nan_true() -> None:
    assert ArrayAllCloseOperator().allclose(
        AllCloseTester(),
        np.array([0.0, 1.0, float("nan")]),
        np.array([0.0, 1.0, float("nan")]),
        equal_nan=True,
    )


@numpy_available
def test_array_allclose_operator_allclose_false_nan_equal_nan_false() -> None:
    assert not ArrayAllCloseOperator().allclose(
        AllCloseTester(),
        np.array([0.0, 1.0, float("nan")]),
        np.array([0.0, 1.0, float("nan")]),
    )


@numpy_available
def test_array_allclose_operator_allclose_false_different_dtype() -> None:
    assert not ArrayAllCloseOperator().allclose(
        AllCloseTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


@numpy_available
def test_array_allclose_operator_allclose_false_different_dtype_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3), dtype=float),
            object2=np.ones((2, 3), dtype=int),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray data types are different:")


@numpy_available
def test_array_allclose_operator_allclose_false_different_shape() -> None:
    assert not ArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), np.zeros((6,)))


@numpy_available
def test_array_allclose_operator_allclose_false_different_shape_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((6,)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray shapes are different:")


@numpy_available
@pytest.mark.parametrize(
    "array", [np.zeros((2, 3)), np.full((2, 3), 1.0 + 1e-7), np.full((2, 3), 1.0 - 1e-7)]
)
def test_array_allclose_operator_allclose_false_different_value(array: np.ndarray) -> None:
    assert not ArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array, rtol=0)


@numpy_available
def test_array_allclose_operator_allclose_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarrays are different")


@numpy_available
def test_array_allclose_operator_allclose_false_different_type() -> None:
    assert not ArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), 42)


@numpy_available
def test_array_allclose_operator_allclose_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=np.ones((2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a numpy.ndarray:")


@numpy_available
@pytest.mark.parametrize(
    ("array", "atol"),
    [(np.full((2, 3), 1.5), 1), (np.full((2, 3), 1.05), 1e-1), (np.full((2, 3), 1.005), 1e-2)],
)
def test_array_allclose_operator_allclose_true_atol(array: np.ndarray, atol: float) -> None:
    assert ArrayAllCloseOperator().allclose(
        AllCloseTester(), np.ones((2, 3)), array, atol=atol, rtol=0
    )


@numpy_available
@pytest.mark.parametrize(
    ("array", "rtol"),
    [(np.full((2, 3), 1.5), 1), (np.full((2, 3), 1.05), 1e-1), (np.full((2, 3), 1.005), 1e-2)],
)
def test_array_allclose_operator_allclose_true_rtol(array: np.ndarray, rtol: float) -> None:
    assert ArrayAllCloseOperator().allclose(AllCloseTester(), np.ones((2, 3)), array, rtol=rtol)


@numpy_available
@pytest.mark.parametrize("check_dtype", [True, False])
def test_array_allclose_operator_clone(check_dtype: bool) -> None:
    op = ArrayAllCloseOperator(check_dtype)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@numpy_available
def test_array_allclose_operator_no_numpy() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args, **kwargs: False):
        with pytest.raises(RuntimeError, match="`numpy` package is required but not installed."):
            ArrayAllCloseOperator()


###########################################
#     Tests for ArrayEqualityOperator     #
###########################################


@numpy_available
def test_objects_are_equal_array() -> None:
    assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)))


@numpy_available
def test_array_equality_operator_str() -> None:
    assert str(ArrayEqualityOperator()).startswith("ArrayEqualityOperator(")


@numpy_available
def test_array_equality_operator__eq__true() -> None:
    assert ArrayEqualityOperator() == ArrayEqualityOperator()


@numpy_available
def test_array_equality_operator__eq__false_different_check_dtype() -> None:
    assert ArrayEqualityOperator(check_dtype=True) != ArrayEqualityOperator(check_dtype=False)


@numpy_available
def test_array_equality_operator__eq__false_different_type() -> None:
    assert ArrayEqualityOperator() != 123


@numpy_available
@pytest.mark.parametrize("check_dtype", [True, False])
def test_array_equality_operator_clone(check_dtype: bool) -> None:
    op = ArrayEqualityOperator(check_dtype)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@numpy_available
def test_array_equality_operator_equal_true() -> None:
    assert ArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.ones((2, 3)))


@numpy_available
def test_array_equality_operator_equal_true_same_object() -> None:
    array = np.ones((2, 3))
    assert ArrayEqualityOperator().equal(EqualityTester(), array, array)


@numpy_available
def test_array_equality_operator_equal_true_check_dtype_false() -> None:
    assert ArrayEqualityOperator(check_dtype=False).equal(
        EqualityTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


@numpy_available
def test_array_equality_operator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert ArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


@numpy_available
def test_array_equality_operator_equal_false_different_dtype() -> None:
    assert not ArrayEqualityOperator().equal(
        EqualityTester(), np.ones((2, 3), dtype=float), np.ones((2, 3), dtype=int)
    )


@numpy_available
def test_array_equality_operator_equal_false_different_dtype_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3), dtype=float),
            object2=np.ones((2, 3), dtype=int),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray data types are different:")


@numpy_available
def test_array_equality_operator_equal_false_different_shape() -> None:
    assert not ArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.zeros((6,)))


@numpy_available
def test_array_equality_operator_equal_false_different_shape_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((6,)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarray shapes are different:")


@numpy_available
def test_array_equality_operator_equal_false_different_value() -> None:
    assert not ArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), np.zeros((2, 3)))


@numpy_available
def test_array_equality_operator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=np.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("numpy.ndarrays are different")


@numpy_available
def test_array_equality_operator_equal_false_different_type() -> None:
    assert not ArrayEqualityOperator().equal(EqualityTester(), np.ones((2, 3)), 42)


@numpy_available
def test_array_equality_operator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not ArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=np.ones((2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a numpy.ndarray:")


@numpy_available
def test_array_equality_operator_no_numpy() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args, **kwargs: False):
        with pytest.raises(RuntimeError, match="`numpy` package is required but not installed."):
            ArrayEqualityOperator()


##########################################
#     Tests for get_mapping_allclose     #
##########################################


@numpy_available
def test_get_mapping_allclose() -> None:
    mapping = get_mapping_allclose()
    assert len(mapping) == 1
    assert isinstance(mapping[np.ndarray], ArrayAllCloseOperator)


def test_get_mapping_allclose_no_numpy() -> None:
    with patch("coola.comparators.numpy_.is_numpy_available", lambda *args, **kwargs: False):
        assert get_mapping_allclose() == {}


##########################################
#     Tests for get_mapping_equality     #
##########################################


@numpy_available
def test_get_mapping_equality() -> None:
    mapping = get_mapping_equality()
    assert len(mapping) == 1
    assert isinstance(mapping[np.ndarray], ArrayEqualityOperator)


def test_get_mapping_equality_no_numpy() -> None:
    with patch("coola.comparators.numpy_.is_numpy_available", lambda *args, **kwargs: False):
        assert get_mapping_equality() == {}

from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import AllCloseTester, objects_are_allclose, objects_are_equal
from coola.comparators import JaxArrayAllCloseOperator, JaxArrayEqualityOperator
from coola.comparators.jax_ import (
    get_array_impl_class,
    get_mapping_allclose,
    get_mapping_equality,
)
from coola.testers import EqualityTester
from coola.testing import jax_available
from coola.utils.imports import is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:
    jnp = Mock()


@jax_available
def test_allclose_tester_registry() -> None:
    assert isinstance(AllCloseTester.registry[jnp.ndarray], JaxArrayAllCloseOperator)


@jax_available
def test_equality_tester_registry() -> None:
    assert isinstance(EqualityTester.registry[jnp.ndarray], JaxArrayEqualityOperator)


##############################################
#     Tests for JaxArrayAllCloseOperator     #
##############################################


@jax_available
def test_objects_are_allclose_array() -> None:
    assert objects_are_allclose(jnp.ones((2, 3)), jnp.ones((2, 3)))


@jax_available
def test_jax_darray_allclose_operator_str() -> None:
    assert str(JaxArrayAllCloseOperator()).startswith("JaxArrayAllCloseOperator(")


@jax_available
def test_jax_darray_allclose_operator__eq__true() -> None:
    assert JaxArrayAllCloseOperator() == JaxArrayAllCloseOperator()


@jax_available
def test_jax_darray_allclose_operator__eq__false_different_check_dtype() -> None:
    assert JaxArrayAllCloseOperator(check_dtype=True) != JaxArrayAllCloseOperator(check_dtype=False)


@jax_available
def test_jax_darray_allclose_operator__eq__false_different_type() -> None:
    assert JaxArrayAllCloseOperator() != 123


@jax_available
@pytest.mark.parametrize(
    "array", [jnp.ones((2, 3)), jnp.full((2, 3), 1.0 + 1e-9), jnp.full((2, 3), 1.0 - 1e-9)]
)
def test_jax_darray_allclose_operator_allclose_true(array: jnp.ndarray) -> None:
    assert JaxArrayAllCloseOperator().allclose(AllCloseTester(), jnp.ones((2, 3)), array)


@jax_available
def test_jax_darray_allclose_operator_allclose_true_same_object() -> None:
    array = jnp.ones((2, 3))
    assert JaxArrayAllCloseOperator().allclose(AllCloseTester(), array, array)


@jax_available
def test_jax_darray_allclose_operator_allclose_true_check_dtype_false() -> None:
    assert JaxArrayAllCloseOperator(check_dtype=False).allclose(
        AllCloseTester(),
        jnp.ones((2, 3), dtype=float),
        jnp.ones((2, 3), dtype=int),
    )


@jax_available
def test_jax_darray_allclose_operator_allclose_true_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert JaxArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=jnp.ones((2, 3)),
            object2=jnp.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


@jax_available
def test_jax_darray_allclose_operator_allclose_true_nan_equal_nan_true() -> None:
    assert JaxArrayAllCloseOperator().allclose(
        AllCloseTester(),
        jnp.array([0.0, 1.0, float("nan")]),
        jnp.array([0.0, 1.0, float("nan")]),
        equal_nan=True,
    )


@jax_available
def test_jax_darray_allclose_operator_allclose_false_nan_equal_nan_false() -> None:
    assert not JaxArrayAllCloseOperator().allclose(
        AllCloseTester(),
        jnp.array([0.0, 1.0, float("nan")]),
        jnp.array([0.0, 1.0, float("nan")]),
    )


@jax_available
def test_jax_darray_allclose_operator_allclose_false_different_dtype() -> None:
    assert not JaxArrayAllCloseOperator().allclose(
        AllCloseTester(), jnp.ones((2, 3), dtype=float), jnp.ones((2, 3), dtype=int)
    )


@jax_available
def test_jax_darray_allclose_operator_allclose_false_different_dtype_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not JaxArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=jnp.ones((2, 3), dtype=float),
            object2=jnp.ones((2, 3), dtype=int),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarray data types are different:")


@jax_available
def test_jax_darray_allclose_operator_allclose_false_different_shape() -> None:
    assert not JaxArrayAllCloseOperator().allclose(
        AllCloseTester(), jnp.ones((2, 3)), jnp.zeros((6,))
    )


@jax_available
def test_jax_darray_allclose_operator_allclose_false_different_shape_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not JaxArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=jnp.ones((2, 3)),
            object2=jnp.zeros((6,)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarray shapes are different:")


@jax_available
@pytest.mark.parametrize(
    "array", [jnp.zeros((2, 3)), jnp.full((2, 3), 1.0 + 1e-7), jnp.full((2, 3), 1.0 - 1e-7)]
)
def test_jax_darray_allclose_operator_allclose_false_different_value(array: jnp.ndarray) -> None:
    assert not JaxArrayAllCloseOperator().allclose(
        AllCloseTester(), jnp.ones((2, 3)), array, rtol=0
    )


@jax_available
def test_jax_darray_allclose_operator_allclose_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not JaxArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=jnp.ones((2, 3)),
            object2=jnp.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarrays are different")


@jax_available
def test_jax_darray_allclose_operator_allclose_false_different_type() -> None:
    assert not JaxArrayAllCloseOperator().allclose(AllCloseTester(), jnp.ones((2, 3)), 42)


@jax_available
def test_jax_darray_allclose_operator_allclose_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not JaxArrayAllCloseOperator().allclose(
            tester=AllCloseTester(),
            object1=jnp.ones((2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a jax.numpy.ndarray:")


@jax_available
@pytest.mark.parametrize(
    ("array", "atol"),
    [(jnp.full((2, 3), 1.5), 1), (jnp.full((2, 3), 1.05), 1e-1), (jnp.full((2, 3), 1.005), 1e-2)],
)
def test_jax_darray_allclose_operator_allclose_true_atol(array: jnp.ndarray, atol: float) -> None:
    assert JaxArrayAllCloseOperator().allclose(
        AllCloseTester(), jnp.ones((2, 3)), array, atol=atol, rtol=0
    )


@jax_available
@pytest.mark.parametrize(
    ("array", "rtol"),
    [(jnp.full((2, 3), 1.5), 1), (jnp.full((2, 3), 1.05), 1e-1), (jnp.full((2, 3), 1.005), 1e-2)],
)
def test_jax_darray_allclose_operator_allclose_true_rtol(array: jnp.ndarray, rtol: float) -> None:
    assert JaxArrayAllCloseOperator().allclose(AllCloseTester(), jnp.ones((2, 3)), array, rtol=rtol)


@jax_available
@pytest.mark.parametrize("check_dtype", [True, False])
def test_jax_darray_allclose_operator_clone(check_dtype: bool) -> None:
    op = JaxArrayAllCloseOperator(check_dtype)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@jax_available
def test_jax_darray_allclose_operator_no_jax() -> None:
    with patch(
        "coola.utils.imports.is_jax_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`jax` package is required but not installed."):
        JaxArrayAllCloseOperator()


##############################################
#     Tests for JaxArrayEqualityOperator     #
##############################################


@jax_available
def test_objects_are_equal_array() -> None:
    assert objects_are_equal(jnp.ones((2, 3)), jnp.ones((2, 3)))


@jax_available
def test_jax_array_equality_operator_str() -> None:
    assert str(JaxArrayEqualityOperator()).startswith("JaxArrayEqualityOperator(")


@jax_available
def test_jax_array_equality_operator__eq__true() -> None:
    assert JaxArrayEqualityOperator() == JaxArrayEqualityOperator()


@jax_available
def test_jax_array_equality_operator__eq__false_different_check_dtype() -> None:
    assert JaxArrayEqualityOperator(check_dtype=True) != JaxArrayEqualityOperator(check_dtype=False)


@jax_available
def test_jax_array_equality_operator__eq__false_different_type() -> None:
    assert JaxArrayEqualityOperator() != 123


@jax_available
@pytest.mark.parametrize("check_dtype", [True, False])
def test_jax_array_equality_operator_clone(check_dtype: bool) -> None:
    op = JaxArrayEqualityOperator(check_dtype)
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@jax_available
def test_jax_array_equality_operator_equal_true() -> None:
    assert JaxArrayEqualityOperator().equal(EqualityTester(), jnp.ones((2, 3)), jnp.ones((2, 3)))


@jax_available
def test_jax_array_equality_operator_equal_true_same_object() -> None:
    array = jnp.ones((2, 3))
    assert JaxArrayEqualityOperator().equal(EqualityTester(), array, array)


@jax_available
def test_jax_array_equality_operator_equal_true_check_dtype_false() -> None:
    assert JaxArrayEqualityOperator(check_dtype=False).equal(
        EqualityTester(), jnp.ones((2, 3), dtype=float), jnp.ones((2, 3), dtype=int)
    )


@jax_available
def test_jax_array_equality_operator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert JaxArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=jnp.ones((2, 3)),
            object2=jnp.ones((2, 3)),
            show_difference=True,
        )
        assert not caplog.messages


@jax_available
def test_jax_array_equality_operator_equal_false_different_dtype() -> None:
    assert not JaxArrayEqualityOperator().equal(
        EqualityTester(), jnp.ones((2, 3), dtype=float), jnp.ones((2, 3), dtype=int)
    )


@jax_available
def test_jax_array_equality_operator_equal_false_different_dtype_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not JaxArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=jnp.ones((2, 3), dtype=float),
            object2=jnp.ones((2, 3), dtype=int),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarray data types are different:")


@jax_available
def test_jax_array_equality_operator_equal_false_different_shape() -> None:
    assert not JaxArrayEqualityOperator().equal(EqualityTester(), jnp.ones((2, 3)), jnp.zeros((6,)))


@jax_available
def test_jax_array_equality_operator_equal_false_different_shape_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not JaxArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=jnp.ones((2, 3)),
            object2=jnp.zeros((6,)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarray shapes are different:")


@jax_available
def test_jax_array_equality_operator_equal_false_different_value() -> None:
    assert not JaxArrayEqualityOperator().equal(
        EqualityTester(), jnp.ones((2, 3)), jnp.zeros((2, 3))
    )


@jax_available
def test_jax_array_equality_operator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not JaxArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=jnp.ones((2, 3)),
            object2=jnp.zeros((2, 3)),
            show_difference=True,
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarrays are different")


@jax_available
def test_jax_array_equality_operator_equal_false_different_type() -> None:
    assert not JaxArrayEqualityOperator().equal(EqualityTester(), jnp.ones((2, 3)), 42)


@jax_available
def test_jax_array_equality_operator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert not JaxArrayEqualityOperator().equal(
            tester=EqualityTester(),
            object1=jnp.ones((2, 3)),
            object2=42,
            show_difference=True,
        )
        assert caplog.messages[0].startswith("object2 is not a jax.numpy.ndarray:")


@jax_available
def test_jax_array_equality_operator_no_jax() -> None:
    with patch(
        "coola.utils.imports.is_jax_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`jax` package is required but not installed."):
        JaxArrayEqualityOperator()


##########################################
#     Tests for get_mapping_allclose     #
##########################################


@jax_available
def test_get_mapping_allclose() -> None:
    mapping = get_mapping_allclose()
    assert len(mapping) == 2
    assert isinstance(mapping[jnp.ndarray], JaxArrayAllCloseOperator)
    assert isinstance(mapping[get_array_impl_class()], JaxArrayAllCloseOperator)


def test_get_mapping_allclose_no_jax() -> None:
    with patch("coola.comparators.jax_.is_jax_available", lambda *args, **kwargs: False):
        assert get_mapping_allclose() == {}


##########################################
#     Tests for get_mapping_equality     #
##########################################


@jax_available
def test_get_mapping_equality() -> None:
    mapping = get_mapping_equality()
    assert len(mapping) == 2
    assert isinstance(mapping[jnp.ndarray], JaxArrayEqualityOperator)
    assert isinstance(mapping[get_array_impl_class()], JaxArrayEqualityOperator)


def test_get_mapping_equality_no_jax() -> None:
    with patch("coola.comparators.jax_.is_jax_available", lambda *args, **kwargs: False):
        assert get_mapping_equality() == {}

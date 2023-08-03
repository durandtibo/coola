from __future__ import annotations

import logging
from unittest.mock import Mock, patch

from pytest import LogCaptureFixture, mark, raises

from coola.comparators import JaxArrayEqualityOperator
from coola.comparators.jax_ import get_mapping_allclose, get_mapping_equality
from coola.testers import EqualityTester
from coola.testing import jax_available
from coola.utils.imports import is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:
    jnp = Mock()


##############################################
#     Tests for JaxArrayEqualityOperator     #
##############################################


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
@mark.parametrize("check_dtype", (True, False))
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
def test_jax_array_equality_operator_equal_true_show_difference(caplog: LogCaptureFixture) -> None:
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
    caplog: LogCaptureFixture,
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
    caplog: LogCaptureFixture,
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
    caplog: LogCaptureFixture,
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
    caplog: LogCaptureFixture,
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
    with patch("coola.utils.imports.is_jax_available", lambda *args, **kwargs: False):
        with raises(RuntimeError, match="`jax` package is required but not installed."):
            JaxArrayEqualityOperator()


##########################################
#     Tests for get_mapping_allclose     #
##########################################


# @jax_available
# def test_get_mapping_allclose() -> None:
#     mapping = get_mapping_allclose()
#     assert len(mapping) == 1
#     assert isinstance(mapping[jnp.ndarray], NDArrayAllCloseOperator)


def test_get_mapping_allclose_no_jax() -> None:
    with patch("coola.comparators.jax_.is_jax_available", lambda *args, **kwargs: False):
        assert get_mapping_allclose() == {}


##########################################
#     Tests for get_mapping_equality     #
##########################################


@jax_available
def test_get_mapping_equality() -> None:
    mapping = get_mapping_equality()
    assert len(mapping) == 1
    assert isinstance(mapping[jnp.ndarray], JaxArrayEqualityOperator)


def test_get_mapping_equality_no_jax() -> None:
    with patch("coola.comparators.jax_.is_jax_available", lambda *args, **kwargs: False):
        assert get_mapping_equality() == {}

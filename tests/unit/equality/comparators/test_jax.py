from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.comparators.jax_ import (
    JaxArrayEqualityComparator,
    get_array_impl_class,
    get_type_comparator_mapping,
)
from coola.testers import EqualityTester
from coola.testing import jax_available
from coola.utils.imports import is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:
    np = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


################################################
#     Tests for JaxArrayEqualityComparator     #
################################################


@jax_available
def test_objects_are_equal_array() -> None:
    assert objects_are_equal(jnp.ones((2, 3)), jnp.ones((2, 3)))


@jax_available
def test_jax_array_equality_comparator_str() -> None:
    assert str(JaxArrayEqualityComparator()).startswith("JaxArrayEqualityComparator(")


@jax_available
def test_jax_array_equality_comparator__eq__true() -> None:
    assert JaxArrayEqualityComparator() == JaxArrayEqualityComparator()


@jax_available
def test_jax_array_equality_comparator__eq__false_different_type() -> None:
    assert JaxArrayEqualityComparator() != 123


@jax_available
def test_jax_array_equality_comparator_clone() -> None:
    op = JaxArrayEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@jax_available
def test_jax_array_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    array = jnp.ones((2, 3))
    assert JaxArrayEqualityComparator().equal(array, array, config)


@jax_available
def test_jax_array_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=jnp.ones((2, 3)),
            object2=jnp.ones((2, 3)),
            config=config,
        )
        assert not caplog.messages


@jax_available
def test_jax_array_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=jnp.ones((2, 3)),
            object2=jnp.ones((2, 3)),
            config=config,
        )
        assert not caplog.messages


@jax_available
def test_jax_array_equality_comparator_equal_false_different_dtype(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=jnp.ones(shape=(2, 3), dtype=float),
            object2=jnp.ones(shape=(2, 3), dtype=int),
            config=config,
        )
        assert not caplog.messages


@jax_available
def test_jax_array_equality_comparator_equal_false_different_dtype_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=jnp.ones(shape=(2, 3), dtype=float),
            object2=jnp.ones(shape=(2, 3), dtype=int),
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different data types:")


@jax_available
def test_jax_array_equality_comparator_equal_false_different_shape(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=jnp.ones((2, 3)),
            object2=jnp.zeros((6,)),
            config=config,
        )
        assert not caplog.messages


@jax_available
def test_jax_array_equality_comparator_equal_false_different_shape_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=jnp.ones((2, 3)),
            object2=jnp.zeros((6,)),
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different shapes:")


@jax_available
def test_jax_array_equality_comparator_equal_false_different_value(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=jnp.ones((2, 3)),
            object2=jnp.zeros((2, 3)),
            config=config,
        )
        assert not caplog.messages


@jax_available
def test_jax_array_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=jnp.ones((2, 3)),
            object2=jnp.zeros((2, 3)),
            config=config,
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarrays have different elements:")


@jax_available
def test_jax_array_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=jnp.ones((2, 3)),
            object2=42,
            config=config,
        )
        assert not caplog.messages


@jax_available
def test_jax_array_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=jnp.ones((2, 3)),
            object2=42,
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different types:")


@jax_available
def test_jax_array_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not JaxArrayEqualityComparator().equal(
        object1=jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
        object2=jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
        config=config,
    )


@jax_available
def test_jax_array_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert JaxArrayEqualityComparator().equal(
        object1=jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
        object2=jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
        config=config,
    )


@jax_available
def test_jax_array_equality_comparator_no_jax() -> None:
    with patch(
        "coola.utils.imports.is_jax_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`jax` package is required but not installed."):
        JaxArrayEqualityComparator()


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


@jax_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        jnp.ndarray: JaxArrayEqualityComparator(),
        get_array_impl_class(): JaxArrayEqualityComparator(),
    }


def test_get_type_comparator_mapping_no_jax() -> None:
    with patch("coola.equality.comparators.jax_.is_jax_available", lambda *args, **kwargs: False):
        assert get_type_comparator_mapping() == {}

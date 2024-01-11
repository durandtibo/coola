from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators.jax_ import (
    JaxArrayEqualityComparator,
    get_array_impl_class,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from coola.testing import jax_available
from coola.utils.imports import is_jax_available
from tests.unit.equality.comparators.utils import ExamplePair
from tests.unit.equality.handlers.test_jax import JAX_ARRAY_TOLERANCE

if is_jax_available():
    import jax.numpy as jnp
else:
    jnp = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


JAX_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=jnp.ones(shape=(2, 3), dtype=float), object2=jnp.ones(shape=(2, 3), dtype=float)
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            object1=jnp.ones(shape=(2, 3), dtype=int), object2=jnp.ones(shape=(2, 3), dtype=int)
        ),
        id="int dtype",
    ),
    pytest.param(ExamplePair(object1=jnp.ones(shape=6), object2=jnp.ones(shape=6)), id="1d array"),
    pytest.param(
        ExamplePair(object1=jnp.ones(shape=(2, 3)), object2=jnp.ones(shape=(2, 3))), id="2d array"
    ),
]


JAX_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=jnp.ones(shape=(2, 3), dtype=float),
            object2=jnp.ones(shape=(2, 3), dtype=int),
            expected_message="objects have different data types:",
        ),
        id="different data types",
    ),
    pytest.param(
        ExamplePair(
            object1=jnp.ones(shape=(2, 3)),
            object2=jnp.ones(shape=6),
            expected_message="objects have different shapes:",
        ),
        id="different shapes",
    ),
    pytest.param(
        ExamplePair(
            object1=jnp.ones(shape=(2, 3)),
            object2=jnp.zeros(shape=(2, 3)),
            expected_message="jax.numpy.ndarrays have different elements:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            object1=jnp.ones(shape=(2, 3)),
            object2="meow",
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]


################################################
#     Tests for JaxArrayEqualityComparator     #
################################################


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
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL)
def test_jax_array_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL)
def test_jax_array_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_NOT_EQUAL)
def test_jax_array_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_NOT_EQUAL)
def test_jax_array_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@jax_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_jax_array_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        JaxArrayEqualityComparator().equal(
            object1=jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
            object2=jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
            config=config,
        )
        == equal_nan
    )


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_TOLERANCE)
def test_jax_array_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert JaxArrayEqualityComparator().equal(
        object1=example.object1, object2=example.object2, config=config
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

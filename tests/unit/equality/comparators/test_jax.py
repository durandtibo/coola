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
from coola.testing.fixtures import jax_available
from coola.utils.imports import is_jax_available
from tests.unit.equality.comparators.utils import ExamplePair

if is_jax_available():
    import jax.numpy as jnp
else:
    jnp = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


JAX_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=jnp.ones(shape=(2, 3), dtype=float), expected=jnp.ones(shape=(2, 3), dtype=float)
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=jnp.ones(shape=(2, 3), dtype=int), expected=jnp.ones(shape=(2, 3), dtype=int)
        ),
        id="int dtype",
    ),
    pytest.param(ExamplePair(actual=jnp.ones(shape=6), expected=jnp.ones(shape=6)), id="1d array"),
    pytest.param(
        ExamplePair(actual=jnp.ones(shape=(2, 3)), expected=jnp.ones(shape=(2, 3))), id="2d array"
    ),
]
JAX_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=jnp.ones(shape=(2, 3), dtype=float),
            expected=jnp.ones(shape=(2, 3), dtype=int),
            expected_message="objects have different data types:",
        ),
        id="different data types",
    ),
    pytest.param(
        ExamplePair(
            actual=jnp.ones(shape=(2, 3)),
            expected=jnp.ones(shape=6),
            expected_message="objects have different shapes:",
        ),
        id="different shapes",
    ),
    pytest.param(
        ExamplePair(
            actual=jnp.ones(shape=(2, 3)),
            expected=jnp.zeros(shape=(2, 3)),
            expected_message="jax.numpy.ndarrays have different elements:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=jnp.ones(shape=(2, 3)),
            expected="meow",
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
JAX_ARRAY_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(actual=jnp.ones((2, 3)), expected=jnp.full((2, 3), 1.5), atol=1.0),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(actual=jnp.ones((2, 3)), expected=jnp.full((2, 3), 1.05), atol=0.1),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(actual=jnp.ones((2, 3)), expected=jnp.full((2, 3), 1.005), atol=0.01),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(actual=jnp.ones((2, 3)), expected=jnp.full((2, 3), 1.5), rtol=1.0),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(actual=jnp.ones((2, 3)), expected=jnp.full((2, 3), 1.05), rtol=0.1),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(actual=jnp.ones((2, 3)), expected=jnp.full((2, 3), 1.005), rtol=0.01),
        id="rtol=0.01",
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
def test_jax_array_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL)
def test_jax_array_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = JaxArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
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
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
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
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@jax_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_jax_array_equality_comparator_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert (
        JaxArrayEqualityComparator().equal(
            actual=jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
            expected=jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
            config=config,
        )
        == equal_nan
    )


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL_TOLERANCE)
def test_jax_array_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert JaxArrayEqualityComparator().equal(
        actual=example.actual, expected=example.expected, config=config
    )


@jax_available
def test_jax_array_equality_comparator_no_jax() -> None:
    with (
        patch("coola.utils.imports.is_jax_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'jax' package is required but not installed."),
    ):
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
    with patch("coola.equality.comparators.jax_.is_jax_available", lambda: False):
        assert get_type_comparator_mapping() == {}

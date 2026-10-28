from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from coola.equality import compare
from coola.equality.config import EqualityConfig
from coola.equality.tester import JaxArrayEqualityTester
from coola.equality.tolerance import Tolerance
from coola.testing.marks import jax_available
from coola.utils.imports import MissingPackageError, is_jax_available
from tests.unit.equality.utils import ExamplePair, explaining, messages, run_tester

if is_jax_available():
    import jax.numpy as jnp
else:
    jnp = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


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
            expected_message="values are different:",
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


############################################
#     Tests for JaxArrayEqualityTester     #
############################################


@jax_available
def test_jax_array_equality_tester_repr() -> None:
    assert repr(JaxArrayEqualityTester()).startswith("JaxArrayEqualityTester(")


@jax_available
def test_jax_array_equality_tester_str() -> None:
    assert str(JaxArrayEqualityTester()).startswith("JaxArrayEqualityTester(")


@jax_available
def test_jax_array_equality_tester_equal_true() -> None:
    assert JaxArrayEqualityTester() == JaxArrayEqualityTester()


@jax_available
def test_jax_array_equality_tester_equal_false_different_type() -> None:
    assert JaxArrayEqualityTester() != 123


@jax_available
def test_jax_array_equality_tester_equal_false_different_type_child() -> None:
    class Child(JaxArrayEqualityTester): ...

    assert JaxArrayEqualityTester() != Child()


@jax_available
def test_jax_array_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    array = jnp.ones((2, 3))
    assert run_tester(JaxArrayEqualityTester(), array, array, config)


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL)
def test_jax_array_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = JaxArrayEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL)
def test_jax_array_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = JaxArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL)
def test_jax_array_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = JaxArrayEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL)
def test_jax_array_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = JaxArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_NOT_EQUAL)
def test_jax_array_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = JaxArrayEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_NOT_EQUAL)
def test_jax_array_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = JaxArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_NOT_EQUAL)
def test_jax_array_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = JaxArrayEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_NOT_EQUAL)
def test_jax_array_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = JaxArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@jax_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_jax_array_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            JaxArrayEqualityTester(),
            jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
            jnp.array([0.0, jnp.nan, jnp.nan, 1.2]),
            config,
        )
        == equal_nan
    )


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL_TOLERANCE)
def test_jax_array_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config = replace(config, tolerance=Tolerance(atol=example.atol, rtol=example.rtol))
    assert run_tester(JaxArrayEqualityTester(), example.actual, example.expected, config)


def test_jax_array_equality_tester_no_jax() -> None:
    with (
        patch("coola.utils.imports.optional.is_jax_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'jax' package is required but not installed."),
    ):
        JaxArrayEqualityTester()


#############################################
#     Tests for the location of a difference #
#############################################


@jax_available
def test_jax_array_compare_path_is_the_index() -> None:
    expected = jnp.zeros((2, 3)).at[1, 2].set(5.0)
    assert compare(jnp.zeros((2, 3)), expected).path == (1, 2)


@jax_available
def test_jax_array_compare_path_is_a_tuple_of_ints() -> None:
    expected = jnp.zeros(2).at[1].set(5.0)
    assert type(compare(jnp.zeros(2), expected).path[0]) is int


@jax_available
def test_jax_array_compare_path_of_a_boolean_array() -> None:
    assert compare(jnp.array([True, False]), jnp.array([True, True])).path == (1,)


@jax_available
def test_jax_array_compare_path_with_a_tolerance() -> None:
    assert compare(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.5]), atol=0.1).path == (1,)


@jax_available
def test_jax_array_compare_nan_with_equal_nan() -> None:
    array = jnp.array([1.0, float("nan")])
    assert compare(array, array, equal_nan=True).equal


@jax_available
def test_jax_array_compare_path_nan_without_equal_nan() -> None:
    assert compare(jnp.array([1.0, float("nan")]), jnp.array([1.0, float("nan")])).path == (1,)


@jax_available
def test_jax_array_compare_reason_has_the_two_values() -> None:
    assert (
        compare(jnp.array([1.0, 2.0]), jnp.array([1.0, 3.0])).reason
        == "values are different:\n  actual   : 2.0\n  expected : 3.0"
    )


@jax_available
def test_jax_array_compare_does_not_look_up_the_index_when_not_explaining(
    config: EqualityConfig,
) -> None:
    assert not run_tester(JaxArrayEqualityTester(), jnp.zeros(3), jnp.ones(3), config)

from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.equality._helpers.jax import array_close, array_equal
from coola.equality.config import EqualityConfig
from coola.equality.tolerance import Tolerance
from coola.testing.marks import jax_available
from coola.utils.imports import is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


##########################################
#     Tests for JaxArrayEqualHandler     #
##########################################


@jax_available
def test_array_equal_rtol_scales_with_the_second_array() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert array_equal(jnp.array([100.0]), jnp.array([101.005]), config)


@jax_available
def test_array_equal_rtol_scales_with_the_first_array() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert array_equal(jnp.array([101.005]), jnp.array([100.0]), config)


@jax_available
def test_array_equal_rtol_outside_the_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not array_equal(jnp.array([101.5]), jnp.array([100.0]), config)


@jax_available
def test_array_equal_rtol_outside_the_tolerance_reversed() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not array_equal(jnp.array([100.0]), jnp.array([101.5]), config)


@jax_available
def test_array_equal_rtol_each_element_in_its_own_direction() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert array_equal(jnp.array([100.0, 101.005]), jnp.array([101.005, 100.0]), config)


@jax_available
def test_array_equal_rtol_each_element_in_its_own_direction_reversed() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert array_equal(jnp.array([101.005, 100.0]), jnp.array([100.0, 101.005]), config)


@jax_available
def test_array_equal_rtol_one_element_outside_the_tolerance() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01, atol=0.0))
    assert not array_equal(jnp.array([100.0, 101.005]), jnp.array([101.005, 105.0]), config)


@jax_available
def test_array_equal_different_shape_with_tolerance() -> None:
    assert not array_equal(jnp.ones(2), jnp.ones(3), EqualityConfig(tolerance=Tolerance(atol=1.0)))


@jax_available
def test_array_equal_different_dtype_with_tolerance() -> None:
    assert not array_equal(
        jnp.ones(2, dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.int32),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@jax_available
def test_array_equal_integers_with_tolerance() -> None:
    assert array_equal(
        jnp.array([100, 101]), jnp.array([101, 100]), EqualityConfig(tolerance=Tolerance(rtol=0.02))
    )


@jax_available
def test_array_equal_bool_has_no_atol() -> None:
    assert not array_equal(
        jnp.array([True]), jnp.array([False]), EqualityConfig(tolerance=Tolerance(atol=1.0))
    )


@jax_available
def test_array_equal_bool_equal_with_tolerance() -> None:
    assert array_equal(
        jnp.array([True, False]),
        jnp.array([True, False]),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    )


@jax_available
def test_array_close_exact() -> None:
    assert array_close(jnp.array([1.0, 2.0]), jnp.array([1.0, 3.0]), EqualityConfig()).tolist() == [
        True,
        False,
    ]


@jax_available
def test_array_close_nan_with_equal_nan() -> None:
    array = jnp.array([float("nan")])
    assert array_close(array, array, EqualityConfig(equal_nan=True)).tolist() == [True]


@jax_available
def test_array_close_nan_without_equal_nan() -> None:
    array = jnp.array([float("nan")])
    assert array_close(array, array, EqualityConfig()).tolist() == [False]


@jax_available
def test_array_close_with_tolerance() -> None:
    assert array_close(
        jnp.array([1.0, 2.0]), jnp.array([1.05, 3.0]), EqualityConfig(tolerance=Tolerance(atol=0.1))
    ).tolist() == [True, False]


@jax_available
def test_array_close_booleans_have_no_tolerance() -> None:
    assert array_close(
        jnp.array([True, False]),
        jnp.array([False, False]),
        EqualityConfig(tolerance=Tolerance(atol=1.0)),
    ).tolist() == [False, True]

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, JaxArrayEqualHandler
from coola.testing.fixtures import jax_available
from coola.utils.imports import is_jax_available
from tests.unit.equality.comparators.test_jax import JAX_ARRAY_EQUAL_TOLERANCE

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()


if TYPE_CHECKING:
    from tests.unit.equality.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


##########################################
#     Tests for JaxArrayEqualHandler     #
##########################################


def test_jax_array_equal_handler__eq__true() -> None:
    assert JaxArrayEqualHandler() == JaxArrayEqualHandler()


def test_jax_array_equal_handler__eq__false_different_type() -> None:
    assert JaxArrayEqualHandler() != FalseHandler()


def test_jax_array_equal_handler__eq__false_different_type_child() -> None:
    class Child(JaxArrayEqualHandler): ...

    assert JaxArrayEqualHandler() != Child()


def test_jax_array_equal_handler_repr() -> None:
    assert repr(JaxArrayEqualHandler()).startswith("JaxArrayEqualHandler(")


def test_jax_array_equal_handler_str() -> None:
    assert str(JaxArrayEqualHandler()).startswith("JaxArrayEqualHandler(")


@jax_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (jnp.ones(shape=(2, 3), dtype=float), jnp.ones(shape=(2, 3), dtype=float)),
        (jnp.ones(shape=(2, 3), dtype=int), jnp.ones(shape=(2, 3), dtype=int)),
        (jnp.ones(shape=(2, 3, 4), dtype=bool), jnp.ones(shape=(2, 3, 4), dtype=bool)),
    ],
)
def test_jax_array_equal_handler_handle_true(
    actual: jnp.ndarray, expected: jnp.ndarray, config: EqualityConfig
) -> None:
    assert JaxArrayEqualHandler().handle(actual, expected, config)


@jax_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (jnp.ones(shape=(2, 3)), jnp.ones(shape=(3, 2))),
        (jnp.ones(shape=(2, 3)), jnp.ones(shape=(2, 1))),
        (jnp.ones(shape=(2, 3)), jnp.ones(shape=(2, 3, 1))),
    ],
)
def test_jax_array_equal_handler_handle_false(
    actual: jnp.ndarray, expected: jnp.ndarray, config: EqualityConfig
) -> None:
    assert not JaxArrayEqualHandler().handle(actual, expected, config)


@jax_available
def test_jax_array_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not JaxArrayEqualHandler().handle(
        jnp.array([0.0, jnp.nan, jnp.nan, 1.2]), jnp.array([0.0, jnp.nan, jnp.nan, 1.2]), config
    )


@jax_available
def test_jax_array_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert JaxArrayEqualHandler().handle(
        jnp.array([0.0, jnp.nan, jnp.nan, 1.2]), jnp.array([0.0, jnp.nan, jnp.nan, 1.2]), config
    )


@jax_available
def test_jax_array_equal_handler_handle_false_show_difference(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = JaxArrayEqualHandler()
    with caplog.at_level(logging.INFO):
        assert not handler.handle(
            actual=jnp.ones(shape=(2, 3)), expected=jnp.ones(shape=(3, 2)), config=config
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarrays have different elements:")


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL_TOLERANCE)
def test_jax_array_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert JaxArrayEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_jax_array_equal_handler_set_next_handler() -> None:
    JaxArrayEqualHandler().set_next_handler(FalseHandler())

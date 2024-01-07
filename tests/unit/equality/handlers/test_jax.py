from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola import EqualityTester
from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, JaxArrayEqualHandler
from coola.testing import jax_available
from coola.utils.imports import is_jax_available

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##########################################
#     Tests for JaxArrayEqualHandler     #
##########################################


def test_jax_array_equal_handler_eq_true() -> None:
    assert JaxArrayEqualHandler() == JaxArrayEqualHandler()


def test_jax_array_equal_handler_eq_false() -> None:
    assert JaxArrayEqualHandler() != FalseHandler()


def test_jax_array_equal_handler_str() -> None:
    assert str(JaxArrayEqualHandler()).startswith("JaxArrayEqualHandler(")


@jax_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (jnp.ones(shape=(2, 3), dtype=float), jnp.ones(shape=(2, 3), dtype=float)),
        (jnp.ones(shape=(2, 3), dtype=int), jnp.ones(shape=(2, 3), dtype=int)),
        (jnp.ones(shape=(2, 3, 4), dtype=bool), jnp.ones(shape=(2, 3, 4), dtype=bool)),
    ],
)
def test_jax_array_equal_handler_handle_true(
    object1: jnp.ndarray, object2: jnp.ndarray, config: EqualityConfig
) -> None:
    assert JaxArrayEqualHandler().handle(object1, object2, config)


@jax_available
@pytest.mark.parametrize(
    ("object1", "object2"),
    [
        (jnp.ones(shape=(2, 3)), jnp.ones(shape=(3, 2))),
        (jnp.ones(shape=(2, 3)), jnp.ones(shape=(2, 1))),
        (jnp.ones(shape=(2, 3)), jnp.ones(shape=(2, 3, 1))),
    ],
)
def test_jax_array_equal_handler_handle_false(
    object1: jnp.ndarray, object2: jnp.ndarray, config: EqualityConfig
) -> None:
    assert not JaxArrayEqualHandler().handle(object1, object2, config)


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
            object1=jnp.ones(shape=(2, 3)), object2=jnp.ones(shape=(3, 2)), config=config
        )
        assert caplog.messages[0].startswith("jax.numpy.ndarrays have different elements:")


def test_jax_array_equal_handler_set_next_handler() -> None:
    JaxArrayEqualHandler().set_next_handler(FalseHandler())
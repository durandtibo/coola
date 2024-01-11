from __future__ import annotations

import logging
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.handlers import FalseHandler, JaxArrayEqualHandler
from coola.equality.testers import EqualityTester
from coola.testing import jax_available
from coola.utils.imports import is_jax_available
from tests.unit.equality.comparators.utils import ExamplePair

if is_jax_available():
    import jax.numpy as jnp
else:  # pragma: no cover
    jnp = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


JAX_ARRAY_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(object1=jnp.ones((2, 3)), object2=jnp.full((2, 3), 1.5), atol=1.0),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(object1=jnp.ones((2, 3)), object2=jnp.full((2, 3), 1.05), atol=0.1),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(object1=jnp.ones((2, 3)), object2=jnp.full((2, 3), 1.005), atol=0.01),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(object1=jnp.ones((2, 3)), object2=jnp.full((2, 3), 1.5), rtol=1.0),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(object1=jnp.ones((2, 3)), object2=jnp.full((2, 3), 1.05), rtol=0.1),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(object1=jnp.ones((2, 3)), object2=jnp.full((2, 3), 1.005), rtol=0.01),
        id="rtol=0.01",
    ),
]


##########################################
#     Tests for JaxArrayEqualHandler     #
##########################################


def test_jax_array_equal_handler_eq_true() -> None:
    assert JaxArrayEqualHandler() == JaxArrayEqualHandler()


def test_jax_array_equal_handler_eq_false() -> None:
    assert JaxArrayEqualHandler() != FalseHandler()


def test_jax_array_equal_handler_repr() -> None:
    assert repr(JaxArrayEqualHandler()).startswith("JaxArrayEqualHandler(")


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


@jax_available
@pytest.mark.parametrize("example", JAX_ARRAY_EQUAL_TOLERANCE)
def test_jax_array_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert JaxArrayEqualHandler().handle(
        object1=example.object1, object2=example.object2, config=config
    )


def test_jax_array_equal_handler_set_next_handler() -> None:
    JaxArrayEqualHandler().set_next_handler(FalseHandler())

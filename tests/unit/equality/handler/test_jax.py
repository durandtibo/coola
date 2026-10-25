from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, JaxArrayEqualHandler, TrueHandler
from coola.equality.handler.jax import array_equal
from coola.testing.fixtures import jax_available
from coola.utils.imports import is_jax_available
from tests.unit.equality.tester.test_jax import JAX_ARRAY_EQUAL_TOLERANCE

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


def test_jax_array_equal_handler_repr() -> None:
    assert repr(JaxArrayEqualHandler()) == "JaxArrayEqualHandler()"


def test_jax_array_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(JaxArrayEqualHandler(FalseHandler()))
        == "JaxArrayEqualHandler(next_handler=FalseHandler())"
    )


def test_jax_array_equal_handler_str() -> None:
    assert str(JaxArrayEqualHandler()) == "JaxArrayEqualHandler()"


def test_jax_array_equal_handler_str_with_next_handler() -> None:
    assert str(JaxArrayEqualHandler(FalseHandler())) == "JaxArrayEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(JaxArrayEqualHandler(), JaxArrayEqualHandler(), id="without next handler"),
        pytest.param(
            JaxArrayEqualHandler(FalseHandler()),
            JaxArrayEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_jax_array_equal_true(
    handler1: JaxArrayEqualHandler, handler2: JaxArrayEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            JaxArrayEqualHandler(TrueHandler()),
            JaxArrayEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            JaxArrayEqualHandler(),
            JaxArrayEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            JaxArrayEqualHandler(FalseHandler()),
            JaxArrayEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(JaxArrayEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_jax_array_equal_false(handler1: JaxArrayEqualHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_jax_array_equal_handler_equal_false_different_type_child() -> None:
    class Child(JaxArrayEqualHandler): ...

    assert not JaxArrayEqualHandler().equal(Child())


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
def test_jax_array_equal_handler_handle_false_show_difference(config: EqualityConfig) -> None:
    config.show_difference = True
    assert not JaxArrayEqualHandler().handle(
        actual=jnp.ones(shape=(2, 3)), expected=jnp.ones(shape=(3, 2)), config=config
    )


@jax_available
def test_jax_array_equal_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        JaxArrayEqualHandler().handle(
            actual=jnp.ones(shape=(2, 3)), expected=jnp.ones(shape=(3, 2)), config=config
        )
    assert caplog.messages[0].startswith("jax.numpy.ndarrays are different:")


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
    handler = JaxArrayEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_jax_array_equal_handler_set_next_handler_none() -> None:
    handler = JaxArrayEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_jax_array_equal_handler_set_next_handler_incorrect() -> None:
    handler = JaxArrayEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


@jax_available
def test_array_equal_rtol_scales_with_the_second_array() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert array_equal(jnp.array([100.0]), jnp.array([101.005]), config)


@jax_available
def test_array_equal_rtol_scales_with_the_first_array() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert array_equal(jnp.array([101.005]), jnp.array([100.0]), config)


@jax_available
def test_array_equal_rtol_outside_the_tolerance() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert not array_equal(jnp.array([101.5]), jnp.array([100.0]), config)


@jax_available
def test_array_equal_rtol_outside_the_tolerance_reversed() -> None:
    config = EqualityConfig(rtol=0.01, atol=0.0)
    assert not array_equal(jnp.array([100.0]), jnp.array([101.5]), config)


@jax_available
def test_array_equal_bool_has_no_atol() -> None:
    assert not array_equal(jnp.array([True]), jnp.array([False]), EqualityConfig(atol=1.0))


@jax_available
def test_array_equal_bool_equal_with_tolerance() -> None:
    assert array_equal(jnp.array([True, False]), jnp.array([True, False]), EqualityConfig(atol=1.0))

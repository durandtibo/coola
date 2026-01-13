from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import JaxArrayEqualityTester
from coola.testing.fixtures import jax_available, jax_not_available
from coola.utils.imports import is_jax_available

if is_jax_available():
    import jax.numpy as jnp


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


@jax_available
def test_jax_array_equality_tester_with_jax(config: EqualityConfig) -> None:
    assert JaxArrayEqualityTester().objects_are_equal(
        jnp.ones((2, 3)), jnp.ones((2, 3)), config=config
    )


@jax_not_available
def test_jax_array_equality_tester_without_jax() -> None:
    with pytest.raises(RuntimeError, match=r"'jax' package is required but not installed."):
        JaxArrayEqualityTester()

from __future__ import annotations

import random
from unittest.mock import Mock, patch

import pytest

from coola.random import (
    BaseRandomManager,
    RandomManagerRegistry,
    RandomRandomManager,
)

###########################################
#     Tests for RandomManagerRegistry     #
###########################################


def test_random_manager_registry_init_empty() -> None:
    registry = RandomManagerRegistry()
    assert len(registry._state) == 0


def test_transformer_registry_init_with_registry() -> None:
    manager = RandomRandomManager()
    initial_registry: dict[str, BaseRandomManager] = {"random": manager}
    registry = RandomManagerRegistry(initial_registry)

    assert "random" in registry._state
    assert registry._state["random"] is manager
    # Verify it's a copy
    initial_registry["default"] = RandomRandomManager()
    assert "default" not in registry._state


def test_random_manager_repr() -> None:
    assert repr(RandomManagerRegistry()).startswith("RandomManagerRegistry(")


def test_random_manager_str() -> None:
    assert str(RandomManagerRegistry()).startswith("RandomManagerRegistry(")


def test_random_manager_get_rng_state() -> None:
    rng = RandomManagerRegistry()
    state = rng.get_rng_state()
    assert isinstance(state, dict)


def test_random_manager_manual_seed() -> None:
    rng = RandomManagerRegistry()
    rng.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


def test_random_manager_set_rng_state() -> None:
    rng = RandomManagerRegistry()
    state = rng.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


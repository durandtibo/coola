from __future__ import annotations

import random
from unittest.mock import Mock, patch

import pytest

from coola.random import BaseRandomManager, RandomManager, RandomRandomManager

###################################
#     Tests for RandomManager     #
###################################


def test_random_manager_repr() -> None:
    assert repr(RandomManager()).startswith("RandomManager(")


def test_random_manager_str() -> None:
    assert str(RandomManager()).startswith("RandomManager(")


def test_random_manager_get_rng_state() -> None:
    rng = RandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, dict)


def test_random_manager_manual_seed() -> None:
    rng = RandomManager()
    rng.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


def test_random_manager_set_rng_state() -> None:
    rng = RandomManager()
    state = rng.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


@patch.dict(RandomManager.registry, {}, clear=True)
def test_random_manager_add_manager() -> None:
    assert len(RandomManager.registry) == 0
    RandomManager.add_manager("random", RandomRandomManager())
    assert isinstance(RandomManager.registry["random"], RandomRandomManager)


@patch.dict(RandomManager.registry, {}, clear=True)
def test_random_manager_add_manager_exist_ok_false() -> None:
    assert len(RandomManager.registry) == 0
    RandomManager.add_manager("random", RandomRandomManager())
    with pytest.raises(
        RuntimeError, match="A random manager .* is already registered for the name 'random'."
    ):
        RandomManager.add_manager("random", RandomRandomManager())


@patch.dict(RandomManager.registry, {}, clear=True)
def test_random_manager_add_manager_exist_ok_true() -> None:
    assert len(RandomManager.registry) == 0
    RandomManager.add_manager("random", Mock(spec=BaseRandomManager))
    RandomManager.add_manager("random", RandomRandomManager(), exist_ok=True)
    assert isinstance(RandomManager.registry["random"], RandomRandomManager)


def test_random_manager_has_manager_true() -> None:
    assert RandomManager.has_manager("random")


def test_random_manager_has_manager_false() -> None:
    assert not RandomManager.has_manager("missing")


def test_random_manager_registered_managers() -> None:
    assert len(RandomManager.registry) >= 1
    assert isinstance(RandomManager.registry["random"], RandomRandomManager)

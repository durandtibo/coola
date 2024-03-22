from __future__ import annotations

import random

from coola import objects_are_equal
from coola.random import RandomRandomManager
from coola.random.random_ import get_random_managers

############################################
#     Tests for RandomRandomSeedSetter     #
############################################


def test_random_random_manager_repr() -> None:
    assert repr(RandomRandomManager()).startswith("RandomRandomManager(")


def test_random_random_manager_str() -> None:
    assert str(RandomRandomManager()).startswith("RandomRandomManager(")


def test_random_random_manager_eq_true() -> None:
    assert RandomRandomManager() == RandomRandomManager()


def test_random_random_manager_eq_false() -> None:
    assert RandomRandomManager() != 42


def test_random_random_manager_get_rng_state() -> None:
    rng = RandomRandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, tuple)


def test_random_random_manager_manual_seed() -> None:
    rng = RandomRandomManager()
    rng.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


def test_random_random_manager_set_rng_state() -> None:
    rng = RandomRandomManager()
    state = rng.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


#########################################
#     Tests for get_random_managers     #
#########################################


def test_get_random_managers() -> None:
    assert objects_are_equal(get_random_managers(), {"random": RandomRandomManager()})

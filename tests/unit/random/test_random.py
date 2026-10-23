from __future__ import annotations

import random

from coola.random import RandomRandomManager

############################################
#     Tests for RandomRandomSeedSetter     #
############################################


def test_random_random_manager_repr() -> None:
    assert repr(RandomRandomManager()).startswith("RandomRandomManager(")


def test_random_random_manager_str() -> None:
    assert str(RandomRandomManager()).startswith("RandomRandomManager(")


def test_random_random_manager__eq__true() -> None:
    assert RandomRandomManager() == RandomRandomManager()


def test_random_random_manager__eq__false_different_type() -> None:
    assert RandomRandomManager() != 42


def test_random_random_manager__eq__false_different_type_child() -> None:
    class Child(RandomRandomManager): ...

    assert RandomRandomManager() != Child()


def test_random_random_manager__hash__() -> None:
    assert hash(RandomRandomManager()) == hash(RandomRandomManager())


def test_random_random_manager__hash__different_type_child() -> None:
    class Child(RandomRandomManager): ...

    assert hash(RandomRandomManager()) != hash(Child())


def test_random_random_manager_get_rng_state() -> None:
    rng = RandomRandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, tuple)


def test_random_random_manager_manual_seed_same_seed_same_values() -> None:
    rng = RandomRandomManager()
    rng.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    random.uniform(0, 1)  # noqa: S311
    rng.manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3


def test_random_random_manager_manual_seed_different_values() -> None:
    rng = RandomRandomManager()
    rng.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.manual_seed(42)
    random.uniform(0, 1)  # noqa: S311
    assert x1 != x2


def test_random_random_manager_set_rng_state_restored_values() -> None:
    rng = RandomRandomManager()
    state = rng.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3


def test_random_random_manager_set_rng_state_different_values() -> None:
    rng = RandomRandomManager()
    state = rng.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    random.uniform(0, 1)  # noqa: S311
    assert x1 != x2


def test_random_random_manager_seed_restores_the_state() -> None:
    manager = RandomRandomManager()
    state = random.getstate()
    with manager.seed(42):
        random.random()  # noqa: S311
    assert random.getstate() == state


def test_random_random_manager_seed_is_reproducible() -> None:
    manager = RandomRandomManager()
    with manager.seed(42):
        first = random.random()  # noqa: S311
    with manager.seed(42):
        second = random.random()  # noqa: S311
    assert first == second


def test_random_random_manager_does_not_manage_random_instances() -> None:
    manager = RandomRandomManager()
    generator = random.Random(0)  # noqa: S311
    expected = random.Random(0).random()  # noqa: S311
    with manager.seed(42):
        value = generator.random()
    assert value == expected

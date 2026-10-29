from __future__ import annotations

import random

import pytest

from coola.random import PythonRandomManager

#########################################
#     Tests for PythonRandomManager     #
#########################################


def test_python_random_manager_repr() -> None:
    assert repr(PythonRandomManager()).startswith("PythonRandomManager(")


def test_python_random_manager_str() -> None:
    assert str(PythonRandomManager()).startswith("PythonRandomManager(")


def test_python_random_manager__eq__true() -> None:
    assert PythonRandomManager() == PythonRandomManager()


def test_python_random_manager__eq__false_different_type() -> None:
    assert PythonRandomManager() != 42


def test_python_random_manager__eq__false_different_type_child() -> None:
    class Child(PythonRandomManager): ...

    assert PythonRandomManager() != Child()


def test_python_random_manager__hash__() -> None:
    assert hash(PythonRandomManager()) == hash(PythonRandomManager())


def test_python_random_manager__hash__different_type_child() -> None:
    class Child(PythonRandomManager): ...

    assert hash(PythonRandomManager()) != hash(Child())


def test_python_random_manager_get_rng_state() -> None:
    rng = PythonRandomManager()
    state = rng.get_rng_state()
    assert isinstance(state, tuple)


def test_python_random_manager_manual_seed_same_seed_same_values() -> None:
    rng = PythonRandomManager()
    rng.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    random.uniform(0, 1)  # noqa: S311
    rng.manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3


def test_python_random_manager_manual_seed_different_values() -> None:
    rng = PythonRandomManager()
    rng.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.manual_seed(42)
    random.uniform(0, 1)  # noqa: S311
    assert x1 != x2


def test_python_random_manager_set_rng_state_restored_values() -> None:
    rng = PythonRandomManager()
    state = rng.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3


def test_python_random_manager_set_rng_state_different_values() -> None:
    rng = PythonRandomManager()
    state = rng.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    random.uniform(0, 1)  # noqa: S311
    assert x1 != x2


def test_python_random_manager_seed_restores_the_state() -> None:
    manager = PythonRandomManager()
    state = random.getstate()
    with manager.seed(42):
        random.random()  # noqa: S311
    assert random.getstate() == state


def test_python_random_manager_seed_is_reproducible() -> None:
    manager = PythonRandomManager()
    with manager.seed(42):
        first = random.random()  # noqa: S311
    with manager.seed(42):
        second = random.random()  # noqa: S311
    assert first == second


def test_python_random_manager_does_not_manage_random_instances() -> None:
    manager = PythonRandomManager()
    generator = random.Random(0)  # noqa: S311
    expected = random.Random(0).random()  # noqa: S311
    with manager.seed(42):
        value = generator.random()
    assert value == expected


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_python_random_manager_manual_seed_invalid_value_raises(seed: int) -> None:
    with pytest.raises(ValueError, match=r"seed must be in the range"):
        PythonRandomManager().manual_seed(seed)


@pytest.mark.parametrize("seed", ["a", 1.5, True])
def test_python_random_manager_manual_seed_invalid_type_raises(seed: object) -> None:
    with pytest.raises(TypeError, match=r"seed must be an int"):
        PythonRandomManager().manual_seed(seed)  # ty: ignore[invalid-argument-type]

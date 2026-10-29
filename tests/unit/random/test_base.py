from __future__ import annotations

import contextlib
from typing import Any

import pytest

from coola.random import BaseRandomManager


class CounterRandomManager(BaseRandomManager):
    r"""A random manager with a state that is easy to inspect."""

    def __init__(self) -> None:
        self.value = 0
        self.calls: list[str] = []

    def get_rng_state(self) -> int:
        self.calls.append("get")
        return self.value

    def manual_seed(self, seed: int) -> None:
        self.calls.append(f"seed({seed})")
        self.value = seed

    def set_rng_state(self, state: Any) -> None:
        self.calls.append(f"set({state})")
        self.value = state


#######################################
#     Tests for BaseRandomManager     #
#######################################


def test_base_random_manager_is_abstract() -> None:
    manager_cls: Any = BaseRandomManager
    with pytest.raises(TypeError, match="abstract"):
        manager_cls()


def test_base_random_manager_seed_sets_the_seed_inside_the_context() -> None:
    manager = CounterRandomManager()
    with manager.seed(42):
        assert manager.value == 42


def test_base_random_manager_seed_restores_the_state() -> None:
    manager = CounterRandomManager()
    manager.value = 7
    with manager.seed(42):
        pass
    assert manager.value == 7


def test_base_random_manager_seed_calls_order() -> None:
    manager = CounterRandomManager()
    with manager.seed(42):
        pass
    assert manager.calls == ["get", "seed(42)", "set(0)"]


def test_base_random_manager_seed_raises_the_exception() -> None:
    manager = CounterRandomManager()
    with pytest.raises(RuntimeError, match="boom"), manager.seed(42):  # noqa: PT012
        msg = "boom"
        raise RuntimeError(msg)


def test_base_random_manager_seed_restores_the_state_with_exception() -> None:
    manager = CounterRandomManager()
    manager.value = 7
    with contextlib.suppress(RuntimeError), manager.seed(42):
        msg = "boom"
        raise RuntimeError(msg)
    assert manager.value == 7


class FailingManager(CounterRandomManager):
    def manual_seed(self, seed: int) -> None:  # noqa: ARG002
        self.value = -1
        msg = "invalid seed"
        raise ValueError(msg)


def test_base_random_manager_seed_raises_the_error_if_seeding_fails() -> None:
    manager = FailingManager()
    with pytest.raises(ValueError, match="invalid seed"), manager.seed(42):
        pass  # pragma: no cover


def test_base_random_manager_seed_restores_the_state_if_seeding_fails() -> None:
    manager = FailingManager()
    manager.value = 7
    with contextlib.suppress(ValueError), manager.seed(42):
        pass  # pragma: no cover
    assert manager.value == 7


def test_base_random_manager_seed_can_be_nested() -> None:
    manager = CounterRandomManager()
    with manager.seed(1), manager.seed(2):
        assert manager.value == 2


def test_base_random_manager_seed_nested_restores_the_outer_seed() -> None:
    manager = CounterRandomManager()
    with manager.seed(1):
        with manager.seed(2):
            pass
        assert manager.value == 1


def test_base_random_manager_seed_nested_restores_the_initial_state() -> None:
    manager = CounterRandomManager()
    with manager.seed(1), manager.seed(2):
        pass
    assert manager.value == 0


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_base_random_manager_seed_invalid_value_raises(seed: int) -> None:
    with (
        pytest.raises(ValueError, match=r"seed must be in the range"),
        CounterRandomManager().seed(seed),
    ):
        pass


@pytest.mark.parametrize("seed", ["a", 1.5, True])
def test_base_random_manager_seed_invalid_type_raises(seed: Any) -> None:
    with (
        pytest.raises(TypeError, match=r"seed must be an int"),
        CounterRandomManager().seed(seed),
    ):
        pass


def test_base_random_manager_seed_invalid_seed_does_not_touch_the_manager() -> None:
    manager = CounterRandomManager()
    with contextlib.suppress(ValueError), manager.seed(-1):
        pass
    assert manager.calls == []

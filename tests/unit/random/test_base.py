from __future__ import annotations

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


##################################
#     Tests for BaseRandomManager #
##################################


def test_base_random_manager_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseRandomManager()  # type: ignore[abstract]


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


def test_base_random_manager_seed_restores_the_state_with_exception() -> None:
    manager = CounterRandomManager()
    manager.value = 7
    with pytest.raises(RuntimeError, match="boom"), manager.seed(42):  # noqa: PT012
        msg = "boom"
        raise RuntimeError(msg)
    assert manager.value == 7


def test_base_random_manager_seed_restores_the_state_if_seeding_fails() -> None:
    class FailingManager(CounterRandomManager):
        def manual_seed(self, seed: int) -> None:  # noqa: ARG002
            self.value = -1
            msg = "invalid seed"
            raise ValueError(msg)

    manager = FailingManager()
    manager.value = 7
    with pytest.raises(ValueError, match="invalid seed"), manager.seed(42):
        pass  # pragma: no cover
    assert manager.value == 7


def test_base_random_manager_seed_can_be_nested() -> None:
    manager = CounterRandomManager()
    with manager.seed(1):
        with manager.seed(2):
            assert manager.value == 2
        assert manager.value == 1
    assert manager.value == 0

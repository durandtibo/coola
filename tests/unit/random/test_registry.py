from __future__ import annotations

import contextlib
import random
from typing import Any

import pytest

from coola.random import (
    BaseRandomManager,
    PythonRandomManager,
    RandomManagerRegistry,
)
from coola.registry import AlreadyRegisteredError, KeyNotRegisteredError


class CounterRandomManager(BaseRandomManager):
    r"""A random manager with a state that is easy to inspect."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def get_rng_state(self) -> int:
        return self.value

    def manual_seed(self, seed: int) -> None:
        self.value = seed

    def set_rng_state(self, state: Any) -> None:
        self.value = state


class FailingRandomManager(CounterRandomManager):
    r"""A random manager that fails to restore its state."""

    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def set_rng_state(self, state: Any) -> None:  # noqa: ARG002
        raise self.error


###########################################
#     Tests for RandomManagerRegistry     #
###########################################


def test_random_manager_registry_init_empty() -> None:
    assert len(RandomManagerRegistry()) == 0


def test_random_manager_registry_init_with_managers() -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    assert registry["random"] == PythonRandomManager()


def test_random_manager_registry_init_copies_the_mapping() -> None:
    initial_state: dict[str, BaseRandomManager] = {"random": PythonRandomManager()}
    registry = RandomManagerRegistry(initial_state)
    initial_state["other"] = PythonRandomManager()
    assert "other" not in registry


def test_random_manager_registry_is_a_manager() -> None:
    assert isinstance(RandomManagerRegistry(), BaseRandomManager)


def test_random_manager_registry_repr() -> None:
    assert repr(RandomManagerRegistry({"random": PythonRandomManager()})) == (
        "RandomManagerRegistry(\n  (random): PythonRandomManager()\n)"
    )


def test_random_manager_registry_str() -> None:
    assert str(RandomManagerRegistry()).startswith("RandomManagerRegistry(")


def test_random_manager_registry_register_existing_key_without_exist_ok() -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        registry.register("random", PythonRandomManager())


def test_random_manager_registry_unregister_missing_raises() -> None:
    with pytest.raises(KeyNotRegisteredError, match="not registered"):
        RandomManagerRegistry().unregister("missing")


def test_random_manager_registry_keys_keep_the_registration_order() -> None:
    registry = RandomManagerRegistry({"b": CounterRandomManager(), "a": CounterRandomManager()})
    assert list(registry.keys()) == ["b", "a"]


def test_random_manager_registry_copy_returns_the_same_class() -> None:
    assert type(RandomManagerRegistry().copy()) is RandomManagerRegistry


def test_random_manager_registry_copy_register_does_not_change_the_original() -> None:
    registry = RandomManagerRegistry()
    registry.copy().register("random", PythonRandomManager())
    assert "random" not in registry


def test_random_manager_registry_equal_copy() -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    assert registry.equal(registry.copy())


def test_random_manager_registry_equal_different_keys() -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    assert not registry.equal(RandomManagerRegistry())


def test_random_manager_registry_get_rng_state() -> None:
    registry = RandomManagerRegistry({"a": CounterRandomManager(1), "b": CounterRandomManager(2)})
    assert registry.get_rng_state() == {"a": 1, "b": 2}


def test_random_manager_registry_get_rng_state_empty() -> None:
    assert RandomManagerRegistry().get_rng_state() == {}


def test_random_manager_registry_get_rng_state_nested_registry() -> None:
    inner = RandomManagerRegistry({"a": CounterRandomManager(1)})
    registry = RandomManagerRegistry({"inner": inner, "b": CounterRandomManager(2)})
    assert registry.get_rng_state() == {"inner": {"a": 1}, "b": 2}


def test_random_manager_registry_manual_seed_seeds_every_manager() -> None:
    first, second = CounterRandomManager(), CounterRandomManager()
    managers: dict[str, BaseRandomManager] = {"a": first, "b": second}
    RandomManagerRegistry(managers).manual_seed(42)
    assert [first.value, second.value] == [42, 42]


def test_random_manager_registry_manual_seed_same_seed_same_values() -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    registry.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    random.uniform(0, 1)  # noqa: S311
    registry.manual_seed(42)
    assert random.uniform(0, 1) == x1  # noqa: S311


@pytest.mark.parametrize("seed", [-1, 2**63, 2**70])
def test_random_manager_registry_manual_seed_out_of_range(seed: int) -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    with pytest.raises(ValueError, match="seed must be in the range"):
        registry.manual_seed(seed)


@pytest.mark.parametrize("seed", [-1, 2**63, 2**70])
def test_random_manager_registry_manual_seed_out_of_range_does_not_seed(seed: int) -> None:
    random.seed(1)
    state = random.getstate()
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    with contextlib.suppress(ValueError):
        registry.manual_seed(seed)
    assert random.getstate() == state  # nothing was seeded


@pytest.mark.parametrize("seed", [1.5, "1", True, None])
def test_random_manager_registry_manual_seed_not_int(seed: object) -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    with pytest.raises(TypeError, match="seed must be an int"):
        registry.manual_seed(seed)  # ty: ignore[invalid-argument-type]


def test_random_manager_registry_unregistered_manager_is_not_seeded() -> None:
    manager = CounterRandomManager()
    registry = RandomManagerRegistry({"a": manager})
    registry.unregister("a")
    registry.manual_seed(42)
    assert manager.value == 0


def test_random_manager_registry_set_rng_state_restores_every_manager() -> None:
    first, second = CounterRandomManager(1), CounterRandomManager(2)
    managers: dict[str, BaseRandomManager] = {"a": first, "b": second}
    registry = RandomManagerRegistry(managers)
    state = registry.get_rng_state()
    registry.manual_seed(42)
    registry.set_rng_state(state)
    assert [first.value, second.value] == [1, 2]


def test_random_manager_registry_set_rng_state_restored_values() -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    state = registry.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    registry.set_rng_state(state)
    assert random.uniform(0, 1) == x1  # noqa: S311


def test_random_manager_registry_set_rng_state_empty_registry() -> None:
    RandomManagerRegistry().set_rng_state({})


def test_random_manager_registry_set_rng_state_nested_registry() -> None:
    inner = RandomManagerRegistry({"a": CounterRandomManager(1)})
    registry = RandomManagerRegistry({"inner": inner})
    state = registry.get_rng_state()
    registry.manual_seed(42)
    registry.set_rng_state(state)
    assert registry.get_rng_state() == {"inner": {"a": 1}}


def test_random_manager_registry_set_rng_state_missing_key_raises() -> None:
    registry = RandomManagerRegistry({"a": CounterRandomManager(), "b": CounterRandomManager()})
    with pytest.raises(ValueError, match=r"The state has the keys \['a'\]"):
        registry.set_rng_state({"a": 1})


def test_random_manager_registry_set_rng_state_extra_key_raises() -> None:
    registry = RandomManagerRegistry({"a": CounterRandomManager()})
    with pytest.raises(ValueError, match=r"registered keys are \['a'\]"):
        registry.set_rng_state({"a": 1, "c": 3})


def test_random_manager_registry_set_rng_state_empty_state_raises() -> None:
    registry = RandomManagerRegistry({"a": CounterRandomManager()})
    with pytest.raises(ValueError, match="The state has the keys"):
        registry.set_rng_state({})


def test_random_manager_registry_set_rng_state_wrong_keys_restores_nothing() -> None:
    first, second = CounterRandomManager(), CounterRandomManager()
    managers: dict[str, BaseRandomManager] = {"a": first, "b": second}
    registry = RandomManagerRegistry(managers)
    with contextlib.suppress(ValueError):
        registry.set_rng_state({"a": 1})
    assert [first.value, second.value] == [0, 0]


def test_random_manager_registry_set_rng_state_nested_wrong_keys_raises() -> None:
    inner = RandomManagerRegistry({"a": CounterRandomManager()})
    registry = RandomManagerRegistry({"inner": inner})
    with pytest.raises(ValueError, match="The state has the keys"):
        registry.set_rng_state({"inner": {"x": 1}})


def test_random_manager_registry_set_rng_state_restores_the_others_when_one_fails() -> None:
    last = CounterRandomManager()
    registry = RandomManagerRegistry(
        {"failing": FailingRandomManager(RuntimeError("boom")), "last": last}
    )
    with contextlib.suppress(RuntimeError):
        registry.set_rng_state({"failing": 0, "last": 5})
    assert last.value == 5


def test_random_manager_registry_set_rng_state_single_error_is_raised_as_is() -> None:
    registry = RandomManagerRegistry({"failing": FailingRandomManager(RuntimeError("boom"))})
    with pytest.raises(RuntimeError, match="boom"):
        registry.set_rng_state({"failing": 0})


def test_random_manager_registry_set_rng_state_several_errors_raise_an_exception_group() -> None:
    registry = RandomManagerRegistry(
        {
            "a": FailingRandomManager(RuntimeError("first")),
            "b": FailingRandomManager(ValueError("second")),
        }
    )
    with pytest.raises(ExceptionGroup) as exc_info:
        registry.set_rng_state({"a": 0, "b": 0})
    assert [str(e) for e in exc_info.value.exceptions] == ["first", "second"]


def test_random_manager_registry_seed_restores_the_state() -> None:
    manager = CounterRandomManager(7)
    registry = RandomManagerRegistry({"a": manager})
    with registry.seed(42):
        pass
    assert manager.value == 7


def test_random_manager_registry_seed_seeds_inside_the_context() -> None:
    manager = CounterRandomManager(7)
    registry = RandomManagerRegistry({"a": manager})
    with registry.seed(42):
        assert manager.value == 42


def test_random_manager_registry_seed_is_reproducible() -> None:
    registry = RandomManagerRegistry({"random": PythonRandomManager()})
    with registry.seed(42):
        first = random.uniform(0, 1)  # noqa: S311
    with registry.seed(42):
        assert random.uniform(0, 1) == first  # noqa: S311

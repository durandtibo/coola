from __future__ import annotations

import random

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


def test_manager_registry_init_with_registry() -> None:
    manager = RandomRandomManager()
    initial_registry: dict[str, BaseRandomManager] = {"random": manager}
    registry = RandomManagerRegistry(initial_registry)

    assert "random" in registry._state
    assert registry._state["random"] is manager
    # Verify it's a copy
    initial_registry["default"] = RandomRandomManager()
    assert "default" not in registry._state


def test_random_manager_registry_repr() -> None:
    assert repr(RandomManagerRegistry()).startswith("RandomManagerRegistry(")


def test_random_manager_registry_str() -> None:
    assert str(RandomManagerRegistry()).startswith("RandomManagerRegistry(")


def test_random_manager_registry_register_new_key() -> None:
    registry = RandomManagerRegistry()
    manager = RandomRandomManager()
    registry.register("random", manager)
    assert registry.has_manager("random")
    assert registry._state["random"] is manager


def test_random_manager_registry_register_existing_key_without_exist_ok() -> None:
    registry = RandomManagerRegistry()
    manager1 = RandomRandomManager()
    manager2 = RandomRandomManager()
    registry.register("random", manager1)
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register("random", manager2, exist_ok=False)


def test_random_manager_registry_register_existing_key_with_exist_ok() -> None:
    registry = RandomManagerRegistry()
    manager1 = RandomRandomManager()
    manager2 = RandomRandomManager()

    registry.register("random", manager1)
    registry.register("random", manager2, exist_ok=True)

    assert registry._state["random"] is manager2


def test_random_manager_registry_register_many() -> None:
    registry = RandomManagerRegistry()
    registry.register_many(
        {
            "random": RandomRandomManager(),
            "default": RandomRandomManager(),
        }
    )
    assert registry.has_manager("random")
    assert registry.has_manager("default")


def test_random_manager_registry_register_many_with_existing_type() -> None:
    registry = RandomManagerRegistry({"random": RandomRandomManager()})
    managers = {
        "random": RandomRandomManager(),
        "default": RandomRandomManager(),
    }
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many(managers, exist_ok=False)


def test_random_manager_registry_register_many_with_exist_ok() -> None:
    registry = RandomManagerRegistry()
    registry.register("random", RandomRandomManager())

    manager = RandomRandomManager()
    managers = {
        "random": manager,
        "default": RandomRandomManager(),
    }

    registry.register_many(managers, exist_ok=True)
    assert registry._state["random"] is manager


def test_random_manager_registry_has_manager_true() -> None:
    assert RandomManagerRegistry({"random": RandomRandomManager()}).has_manager("random")


def test_random_manager_registry_has_manager_false() -> None:
    assert not RandomManagerRegistry().has_manager("random")


def test_random_manager_registry_get_rng_state() -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager()})
    state = rng.get_rng_state()
    assert isinstance(state, dict)


def test_random_manager_registry_manual_seed() -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager()})
    rng.manual_seed(42)
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.manual_seed(42)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2


def test_random_manager_registry_set_rng_state() -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager()})
    state = rng.get_rng_state()
    x1 = random.uniform(0, 1)  # noqa: S311
    x2 = random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    x3 = random.uniform(0, 1)  # noqa: S311
    assert x1 == x3
    assert x1 != x2

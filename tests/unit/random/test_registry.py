from __future__ import annotations

import random

import pytest

from coola.random import (
    BaseRandomManager,
    RandomManagerRegistry,
    RandomRandomManager,
)
from coola.registry import AlreadyRegisteredError

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
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
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
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
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


def test_random_manager_registry_set_rng_state_unregistered_key() -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager()})
    state = rng.get_rng_state()
    state["torch"] = "fake_state"
    # Should not raise even though "torch" is not registered.
    rng.set_rng_state(state)


def test_random_manager_registry_set_rng_state_unregistered_key_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = RandomManagerRegistry()
    with caplog.at_level("WARNING"):
        rng.set_rng_state({"torch": "fake_state"})
    assert "torch" in caplog.text


def test_random_manager_registry_set_rng_state_empty() -> None:
    rng = RandomManagerRegistry()
    # Should not raise for an empty registry and empty state.
    rng.set_rng_state({})


def test_random_manager_registry_set_rng_state_missing_key_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager(), "other": RandomRandomManager()})
    state = rng.get_rng_state()
    del state["other"]
    with caplog.at_level("WARNING"):
        rng.set_rng_state(state)
    assert "No RNG state for the registered key(s) ['other']" in caplog.text


def test_random_manager_registry_set_rng_state_missing_key_restores_the_other_keys() -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager(), "other": RandomRandomManager()})
    state = rng.get_rng_state()
    del state["other"]
    x1 = random.uniform(0, 1)  # noqa: S311
    rng.set_rng_state(state)
    assert random.uniform(0, 1) == x1  # noqa: S311


def test_random_manager_registry_set_rng_state_all_keys_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager()})
    with caplog.at_level("WARNING"):
        rng.set_rng_state(rng.get_rng_state())
    assert not caplog.records


def test_random_manager_registry_set_rng_state_empty_state_warns_for_registered_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager()})
    with caplog.at_level("WARNING"):
        rng.set_rng_state({})
    assert "['random']" in caplog.text


def test_random_manager_registry_seed() -> None:
    rng = RandomManagerRegistry({"random": RandomRandomManager()})
    state = random.getstate()
    with rng.seed(42):
        first = random.uniform(0, 1)  # noqa: S311
    assert random.getstate() == state
    with rng.seed(42):
        second = random.uniform(0, 1)  # noqa: S311
    assert first == second


def test_random_manager_registry_find_manager() -> None:
    manager = RandomRandomManager()
    registry = RandomManagerRegistry({"random": manager})
    assert registry.find_manager("random") is manager


def test_random_manager_registry_find_manager_missing_raises() -> None:
    registry = RandomManagerRegistry({"random": RandomRandomManager()})
    with pytest.raises(KeyError, match="Key 'torch' is not registered"):
        registry.find_manager("torch")


def test_random_manager_registry_unregister_returns_the_manager() -> None:
    manager = RandomRandomManager()
    registry = RandomManagerRegistry({"random": manager})
    assert registry.unregister("random") is manager
    assert not registry.has_manager("random")


def test_random_manager_registry_unregister_missing_raises() -> None:
    registry = RandomManagerRegistry()
    with pytest.raises(KeyError, match="Key 'random' is not registered"):
        registry.unregister("random")


def test_random_manager_registry_unregistered_manager_is_not_seeded() -> None:
    registry = RandomManagerRegistry({"random": RandomRandomManager()})
    registry.unregister("random")
    state = random.getstate()
    registry.manual_seed(1)
    assert random.getstate() == state


def test_random_manager_registry_unregistered_manager_is_not_in_the_state() -> None:
    registry = RandomManagerRegistry({"random": RandomRandomManager()})
    registry.unregister("random")
    assert registry.get_rng_state() == {}


@pytest.mark.parametrize("seed", [-1, 2**63, 2**70])
def test_random_manager_registry_manual_seed_out_of_range(seed: int) -> None:
    random.seed(1)
    state = random.getstate()
    registry = RandomManagerRegistry({"random": RandomRandomManager()})
    with pytest.raises(ValueError, match="seed must be in the range"):
        registry.manual_seed(seed)
    assert random.getstate() == state  # nothing was seeded


@pytest.mark.parametrize("seed", [1.5, "1", True, None])
def test_random_manager_registry_manual_seed_not_int(seed: object) -> None:
    registry = RandomManagerRegistry({"random": RandomRandomManager()})
    with pytest.raises(TypeError, match="seed must be an int"):
        registry.manual_seed(seed)

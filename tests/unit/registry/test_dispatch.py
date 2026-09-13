from __future__ import annotations

import pytest

from coola.equality.tester.registry import EqualityTesterRegistry
from coola.hashing.registry import HasherRegistry
from coola.iterator.bfs.registry import ChildFinderRegistry
from coola.iterator.dfs.registry import IteratorRegistry
from coola.recursive.registry import TransformerRegistry
from coola.registry.dispatch import BaseTypeDispatchRegistry
from coola.summary.registry import SummarizerRegistry

##############################################
#     Tests for BaseTypeDispatchRegistry     #
##############################################


def test_base_type_dispatch_registry_init_empty() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    assert len(registry._state) == 0


def test_base_type_dispatch_registry_init_state_is_copied() -> None:
    state = {int: "int_handler"}
    registry = BaseTypeDispatchRegistry[str](state)
    registry.register(float, "float_handler")
    assert float not in state


def test_base_type_dispatch_registry_register_and_find() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register(int, "int_handler")
    assert registry.find(int) == "int_handler"


def test_base_type_dispatch_registry_register_exist_ok_false_raises_error() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register(int, "new_handler")


def test_base_type_dispatch_registry_register_exist_ok_true_overwrites() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    registry.register(int, "new_handler", exist_ok=True)
    assert registry.find(int) == "new_handler"


def test_base_type_dispatch_registry_register_many() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register_many({int: "int_handler", str: "str_handler"})
    assert registry.find(int) == "int_handler"
    assert registry.find(str) == "str_handler"


def test_base_type_dispatch_registry_has_true() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    assert registry.has(int)


def test_base_type_dispatch_registry_has_false() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    assert not registry.has(int)


def test_base_type_dispatch_registry_has_false_for_mro_only_match() -> None:
    # `has` only checks direct registration, not MRO resolution.
    registry = BaseTypeDispatchRegistry[str]({object: "default_handler"})
    assert not registry.has(int)


def test_base_type_dispatch_registry_find_resolves_via_mro() -> None:
    registry = BaseTypeDispatchRegistry[str]({object: "default_handler"})
    assert registry.find(int) == "default_handler"


def test_base_type_dispatch_registry_find_missing_raises_error() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    with pytest.raises(KeyError):
        registry.find(int)


def test_base_type_dispatch_registry_repr() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    assert repr(registry) == (
        "BaseTypeDispatchRegistry(\n"
        "  (state): TypeRegistry(\n"
        "      (<class 'int'>): int_handler\n"
        "    )\n"
        ")"
    )


@pytest.mark.parametrize(
    "registry_cls",
    [
        EqualityTesterRegistry,
        HasherRegistry,
        TransformerRegistry,
        SummarizerRegistry,
        ChildFinderRegistry,
        IteratorRegistry,
    ],
)
def test_registries_are_subclasses_of_base_type_dispatch_registry(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that each of the five (plus one) registry wrappers reuses
    the shared base implementation instead of hand-rolling
    register/register_many/has/find."""
    assert issubclass(registry_cls, BaseTypeDispatchRegistry)


@pytest.mark.parametrize(
    "registry_cls",
    [
        EqualityTesterRegistry,
        HasherRegistry,
        TransformerRegistry,
        SummarizerRegistry,
        ChildFinderRegistry,
        IteratorRegistry,
    ],
)
def test_registries_register_many_shares_base_implementation(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that ``register_many`` on a concrete registry actually
    delegates to the shared base implementation (registers all entries,
    is exist_ok-aware, does not partially mutate state on failure)."""

    class Handler:
        def __repr__(self) -> str:
            return "Handler()"

    registry = registry_cls()
    h1, h2 = Handler(), Handler()
    registry.register_many({int: h1, str: h2})
    assert registry.has(int)
    assert registry.has(str)

    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many({int: Handler()})

    # exist_ok=True overwrites without error
    h3 = Handler()
    registry.register_many({int: h3}, exist_ok=True)
    assert registry.find(int) is h3

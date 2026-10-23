from __future__ import annotations

from decimal import Decimal

import pytest

from coola.equality.tester.registry import EqualityTesterRegistry
from coola.hashing.registry import HasherRegistry
from coola.iterator.bfs.registry import ChildFinderRegistry
from coola.iterator.dfs.registry import IteratorRegistry
from coola.recursive.registry import TransformerRegistry
from coola.registry import AlreadyRegisteredError, KeyNotRegisteredError
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
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        registry.register(int, "new_handler")


def test_base_type_dispatch_registry_register_exist_ok_true_overwrites() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    registry.register(int, "new_handler", exist_ok=True)
    assert registry.find(int) == "new_handler"


def test_base_type_dispatch_registry_register_many() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register_many({int: "int_handler", str: "str_handler"})
    assert registry.find(int) == "int_handler"


def test_base_type_dispatch_registry_register_many_registers_all_the_types() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register_many({int: "int_handler", str: "str_handler"})
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


REGISTRY_CLASSES = [
    EqualityTesterRegistry,
    HasherRegistry,
    TransformerRegistry,
    SummarizerRegistry,
    ChildFinderRegistry,
    IteratorRegistry,
]


class Handler:
    def __repr__(self) -> str:
        return "Handler()"


@pytest.mark.parametrize("registry_cls", REGISTRY_CLASSES)
def test_registries_are_subclasses_of_base_type_dispatch_registry(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that each of the five (plus one) registry wrappers reuses
    the shared base implementation instead of hand-rolling
    register/register_many/has/find."""
    assert issubclass(registry_cls, BaseTypeDispatchRegistry)


@pytest.mark.parametrize("registry_cls", REGISTRY_CLASSES)
def test_registries_register_many_shares_base_implementation(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that ``register_many`` on a concrete registry actually
    delegates to the shared base implementation (registers all entries,
    is exist_ok-aware, does not partially mutate state on failure)."""
    registry = registry_cls()
    h1, h2 = Handler(), Handler()
    registry.register_many({int: h1, str: h2})
    assert registry.has(int)


@pytest.mark.parametrize("registry_cls", REGISTRY_CLASSES)
def test_registries_register_many_registers_all_the_entries(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that ``register_many`` on a concrete registry registers the
    second entry."""
    registry = registry_cls()
    registry.register_many({int: Handler(), str: Handler()})
    assert registry.has(str)


@pytest.mark.parametrize("registry_cls", REGISTRY_CLASSES)
def test_registries_register_many_raises_on_duplicate(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that ``register_many`` on a concrete registry raises when a
    type is already registered."""
    registry = registry_cls()
    registry.register_many({int: Handler()})
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        registry.register_many({int: Handler()})


@pytest.mark.parametrize("registry_cls", REGISTRY_CLASSES)
def test_registries_register_many_exist_ok_overwrites(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    """Test that ``register_many`` with ``exist_ok=True`` overwrites
    without error."""
    registry = registry_cls()
    registry.register_many({int: Handler()})
    h3 = Handler()
    registry.register_many({int: h3}, exist_ok=True)
    assert registry.find(int) is h3


def test_base_type_dispatch_registry_len() -> None:
    assert len(BaseTypeDispatchRegistry[str]({int: "a", str: "b"})) == 2


def test_base_type_dispatch_registry_contains_true() -> None:
    assert int in BaseTypeDispatchRegistry[str]({int: "a"})


def test_base_type_dispatch_registry_contains_false() -> None:
    assert str not in BaseTypeDispatchRegistry[str]({int: "a"})


def test_base_type_dispatch_registry_unregister_returns_handler() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "a"})
    assert registry.unregister(int) == "a"


def test_base_type_dispatch_registry_unregister_removes_type() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "a"})
    registry.unregister(int)
    assert not registry.has(int)


def test_base_type_dispatch_registry_unregister_clears_the_cache() -> None:
    registry = BaseTypeDispatchRegistry[str]({object: "obj", int: "a"})
    registry.find(int)
    registry.unregister(int)
    assert registry.find(int) == "obj"


def test_base_type_dispatch_registry_unregister_missing_raises() -> None:
    with pytest.raises(KeyNotRegisteredError, match="not registered"):
        BaseTypeDispatchRegistry[str]().unregister(int)


def test_base_type_dispatch_registry_clear() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "a"})
    registry.clear()
    assert len(registry) == 0


def test_base_type_dispatch_registry_items() -> None:
    assert dict(BaseTypeDispatchRegistry[str]({int: "a"}).items()) == {int: "a"}


def test_base_type_dispatch_registry_equal_true() -> None:
    assert BaseTypeDispatchRegistry[str]({int: "a"}).equal(
        BaseTypeDispatchRegistry[str]({int: "a"})
    )


def test_base_type_dispatch_registry_equal_false_different_handlers() -> None:
    assert not BaseTypeDispatchRegistry[str]({int: "a"}).equal(
        BaseTypeDispatchRegistry[str]({int: "b"})
    )


def test_base_type_dispatch_registry_equal_false_different_type() -> None:
    assert not BaseTypeDispatchRegistry[str]({int: "a"}).equal({int: "a"})


###########################################
#     Tests for the lazy registration     #
###########################################


def test_base_type_dispatch_registry_register_lazy_find_loads_the_handlers() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.find(Decimal) == "decimal"


def test_base_type_dispatch_registry_register_lazy_len_loads_the_handlers() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert len(registry) == 1

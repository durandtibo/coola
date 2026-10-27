from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from decimal import Decimal

import pytest

from coola.equality.tester.registry import EqualityTesterRegistry
from coola.hashing.registry import HasherRegistry
from coola.iterator.registry import ChildFinderRegistry
from coola.recursive.registry import TransformerRegistry
from coola.registry import AlreadyRegisteredError, KeyNotRegisteredError
from coola.registry.dispatch import BaseTypeDispatchRegistry, FallbackMixin
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


def test_base_type_dispatch_registry_copy_returns_the_same_class() -> None:
    assert type(HasherRegistry().copy()) is HasherRegistry


def test_base_type_dispatch_registry_copy_has_the_same_handlers() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler", float: "float_handler"})
    assert dict(registry.copy().items()) == {int: "int_handler", float: "float_handler"}


def test_base_type_dispatch_registry_copy_register_does_not_change_the_original() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    registry.copy().register(float, "float_handler")
    assert not registry.has(float)


def test_base_type_dispatch_registry_copy_unregister_does_not_change_the_original() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    registry.copy().unregister(int)
    assert registry.has(int)


def test_base_type_dispatch_registry_copy_clear_does_not_change_the_original() -> None:
    registry = BaseTypeDispatchRegistry[str]({int: "int_handler"})
    registry.copy().clear()
    assert len(registry) == 1


def test_base_type_dispatch_registry_copy_has_the_lazy_groups() -> None:
    registry = BaseTypeDispatchRegistry[str]()
    registry.register_lazy("decimal", lambda: {Decimal: "decimal"})
    assert registry.copy().find(Decimal) == "decimal"


def test_base_type_dispatch_registry_copy_does_not_copy_the_handlers() -> None:
    handler = object()
    assert BaseTypeDispatchRegistry[object]({int: handler}).copy().find(int) is handler


@pytest.mark.parametrize(
    "registry_cls",
    [
        EqualityTesterRegistry,
        HasherRegistry,
        ChildFinderRegistry,
        TransformerRegistry,
        SummarizerRegistry,
    ],
)
def test_base_type_dispatch_registry_copy_of_a_subclass_has_its_own_state(
    registry_cls: type[BaseTypeDispatchRegistry],
) -> None:
    registry = registry_cls()
    assert registry.copy()._state is not registry._state


@dataclass
class Point:
    x: int


###################################
#     Tests for FallbackMixin     #
###################################


class _FallbackRegistry(FallbackMixin[str], BaseTypeDispatchRegistry[str]):
    pass


def test_fallback_mixin_register_fallback_find() -> None:
    registry = _FallbackRegistry({object: "object"})
    registry.register_fallback(is_dataclass, "dataclass")
    assert registry.find(Point) == "dataclass"


def test_fallback_mixin_register_fallback_duplicate_raises() -> None:
    registry = _FallbackRegistry()
    registry.register_fallback(is_dataclass, "dataclass")
    with pytest.raises(AlreadyRegisteredError, match="A fallback is already registered"):
        registry.register_fallback(is_dataclass, "other")


def test_fallback_mixin_register_fallback_exist_ok() -> None:
    registry = _FallbackRegistry()
    registry.register_fallback(is_dataclass, "dataclass")
    registry.register_fallback(is_dataclass, "other", exist_ok=True)
    assert list(registry.fallbacks()) == [(is_dataclass, "other")]


def test_fallback_mixin_unregister_fallback_returns_the_handler() -> None:
    registry = _FallbackRegistry()
    registry.register_fallback(is_dataclass, "dataclass")
    assert registry.unregister_fallback(is_dataclass) == "dataclass"


def test_fallback_mixin_unregister_fallback_unknown_raises() -> None:
    with pytest.raises(KeyNotRegisteredError, match="No fallback is registered"):
        _FallbackRegistry().unregister_fallback(is_dataclass)


def test_fallback_mixin_copy_has_the_fallbacks() -> None:
    registry = _FallbackRegistry()
    registry.register_fallback(is_dataclass, "dataclass")
    assert list(registry.copy().fallbacks()) == [(is_dataclass, "dataclass")]


def test_fallback_mixin_clear_removes_the_fallbacks() -> None:
    registry = _FallbackRegistry()
    registry.register_fallback(is_dataclass, "dataclass")
    registry.clear()
    assert list(registry.fallbacks()) == []

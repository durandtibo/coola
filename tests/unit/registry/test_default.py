from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

from coola.registry.default import (
    make_default_registry_singleton,
    reset_default_registries,
    temporary_default_registries,
)
from coola.registry.plugins import (
    get_loaded_plugins,
)
from tests.unit.registry.helpers import FakeEntryPoint, noop, patch_entry_points

if TYPE_CHECKING:
    from coola.utils.singleton import LazySingleton

############################################
#     Tests for the default registries     #
############################################


def test_make_default_registry_singleton_plugin_group_is_keyword_only() -> None:
    function: Any = make_default_registry_singleton
    with pytest.raises(TypeError, match="positional arguments"):
        function(dict, lambda _: None, "grp")


def test_make_default_registry_singleton_is_lazy() -> None:
    calls = []

    def register(_registry: dict) -> None:
        calls.append(1)

    make_default_registry_singleton(dict, register)
    assert calls == []


def test_make_default_registry_singleton_populates_the_registry() -> None:
    def register(registry: dict) -> None:
        registry["a"] = 1

    assert make_default_registry_singleton(dict, register).get() == {"a": 1}


def test_make_default_registry_singleton_returns_the_same_registry() -> None:
    singleton = make_default_registry_singleton(dict, lambda _: None)
    assert singleton.get() is singleton.get()


def test_make_default_registry_singleton_builds_and_populates_once() -> None:
    calls = []

    def register(_registry: dict) -> None:
        calls.append(1)

    singleton = make_default_registry_singleton(dict, register)
    singleton.get()
    singleton.get()
    assert calls == [1]


def test_make_default_registry_singleton_concurrent_get_builds_once() -> None:
    calls = []

    def register(_registry: dict) -> None:
        calls.append(1)

    singleton = make_default_registry_singleton(dict, register)
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: singleton.get(), range(32)))
    assert calls == [1]


def test_make_default_registry_singleton_concurrent_get_returns_the_same_registry() -> None:
    singleton = make_default_registry_singleton(dict, lambda _: None)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: singleton.get(), range(32)))
    assert all(r is results[0] for r in results)


def test_make_default_registry_singleton_loads_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", lambda r: r.update(p=1))])
    holder = make_default_registry_singleton(dict, lambda r: r.update(d=0), plugin_group="grp")
    assert holder.get() == {"d": 0, "p": 1}


def make_singleton_with_a_reentrant_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> LazySingleton[dict[str, int]]:
    holder: dict[str, LazySingleton[dict[str, int]]] = {}

    def bad_plugin(registry: Any) -> None:  # noqa: ARG001
        holder["singleton"].get()

    def good_plugin(registry: dict[str, int]) -> None:
        registry["good"] = 1

    patch_entry_points(
        monkeypatch, [FakeEntryPoint("bad", bad_plugin), FakeEntryPoint("good", good_plugin)]
    )
    holder["singleton"] = make_default_registry_singleton(dict, lambda _: None, plugin_group="grp")
    return holder["singleton"]


def test_default_registry_singleton_plugin_cannot_request_the_registry_being_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    singleton = make_singleton_with_a_reentrant_plugin(monkeypatch)
    with pytest.warns(RuntimeWarning, match="requested while it is being built"):
        singleton.get()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_default_registry_singleton_loads_the_other_plugins_after_a_reentrant_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    singleton = make_singleton_with_a_reentrant_plugin(monkeypatch)
    registry = singleton.get()
    assert registry == {"good": 1}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_default_registry_singleton_returns_the_same_registry_after_a_reentrant_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    singleton = make_singleton_with_a_reentrant_plugin(monkeypatch)
    registry = singleton.get()
    assert singleton.get() is registry


def test_reset_default_registries_resets_the_singletons() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    registry = singleton.get()
    reset_default_registries()
    assert singleton.get() is not registry


def test_reset_default_registries_drops_the_changes() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    singleton.get().clear()
    reset_default_registries()
    assert singleton.get() == {"a": 1}


def test_reset_default_registries_does_not_change_the_previous_registry() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    registry = singleton.get()
    reset_default_registries()
    assert registry == {"a": 1}


def test_reset_default_registries_does_not_build_the_registries() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    reset_default_registries()
    assert not singleton.is_built()


def test_temporary_default_registries_uses_a_copy() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    registry = singleton.get()
    with temporary_default_registries():
        assert singleton.get() is not registry


def test_temporary_default_registries_copy_has_the_entries() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    singleton.get()
    with temporary_default_registries():
        assert singleton.get() == {"a": 1}


def test_temporary_default_registries_restores_the_registry() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    registry = singleton.get()
    with temporary_default_registries():
        singleton.get().clear()
    assert singleton.get() is registry


def test_temporary_default_registries_does_not_change_the_original() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    singleton.get()
    with temporary_default_registries():
        singleton.get().clear()
    assert singleton.get() == {"a": 1}


def test_temporary_default_registries_forgets_the_registry_built_inside() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    with temporary_default_registries():
        singleton.get()
    assert not singleton.is_built()


def fail_in_scope(singleton: LazySingleton[dict]) -> None:
    with temporary_default_registries():
        singleton.get().clear()
        msg = "boom"
        raise RuntimeError(msg)


def test_temporary_default_registries_restores_after_an_error() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    registry = singleton.get()
    with pytest.raises(RuntimeError, match="boom"):
        fail_in_scope(singleton)
    assert singleton.get() is registry


def test_temporary_default_registries_nested() -> None:
    singleton = make_default_registry_singleton(dict, lambda r: r.update(a=1))
    registry = singleton.get()
    with temporary_default_registries(), temporary_default_registries():
        singleton.get().clear()
    assert singleton.get() is registry


def test_get_loaded_plugins_after_a_default_registry_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", lambda r: r.update(p=1))])
    make_default_registry_singleton(dict, noop, plugin_group="grp").get()
    assert [plugin.name for plugin in get_loaded_plugins()] == ["a"]

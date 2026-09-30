from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from coola.utils import singleton
from coola.utils.singleton import (
    LazySingleton,
    load_registry_plugins,
    make_default_registry_singleton,
)

if TYPE_CHECKING:
    from collections.abc import Callable

###################################
#     Tests for LazySingleton     #
###################################


def test_lazy_singleton_get_returns_value_from_factory() -> None:
    singleton = LazySingleton(lambda: 42)
    assert singleton.get() == 42


def test_lazy_singleton_does_not_call_factory_until_get() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        return 42

    LazySingleton(factory)
    assert calls == []


def test_lazy_singleton_calls_factory_only_once() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        return 42

    singleton = LazySingleton(factory)
    singleton.get()
    singleton.get()
    singleton.get()
    assert len(calls) == 1


def test_lazy_singleton_returns_same_instance() -> None:
    singleton = LazySingleton(object)
    assert singleton.get() is singleton.get()


def test_lazy_singleton_returns_mutated_instance() -> None:
    singleton = LazySingleton(list)
    instance1 = singleton.get()
    instance1.append(1)
    instance2 = singleton.get()
    assert instance1 is instance2
    assert instance2 == [1]


def test_lazy_singleton_independent_instances_have_independent_state() -> None:
    singleton1 = LazySingleton(list)
    singleton2 = LazySingleton(list)
    singleton1.get().append(1)
    assert singleton1.get() == [1]
    assert singleton2.get() == []


def test_lazy_singleton_falsy_value_is_cached_correctly() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        return 0

    singleton = LazySingleton(factory)
    assert singleton.get() == 0
    assert singleton.get() == 0
    assert len(calls) == 1


def test_lazy_singleton_none_value_calls_factory_every_time() -> None:
    # A factory returning None cannot be cached because the implementation
    # uses `self._instance is None` as the "not built yet" sentinel.
    calls = []

    def factory() -> None:
        calls.append(1)
        return

    singleton = LazySingleton(factory)
    singleton.get()
    singleton.get()
    assert len(calls) == 2


def test_lazy_singleton_propagates_factory_exception() -> None:
    def factory() -> int:
        msg = "boom"
        raise RuntimeError(msg)

    singleton = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="boom"):
        singleton.get()


def test_lazy_singleton_recovers_after_factory_exception() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        if len(calls) == 1:
            msg = "boom"
            raise RuntimeError(msg)
        return 42

    singleton = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="boom"):
        singleton.get()
    assert singleton.get() == 42
    assert len(calls) == 2


def test_lazy_singleton_get_skips_factory_when_instance_set_during_lock_acquisition() -> None:
    # Covers the double-checked-locking branch: if the instance was built by
    # another caller while this one was waiting to acquire the lock, the
    # factory must not be called again once the lock is held.

    class SettingLock:
        def __init__(self, singleton: LazySingleton[str]) -> None:
            self._singleton = singleton

        def __enter__(self) -> None:
            self._singleton._instance = "built-by-another-caller"

        def __exit__(self, *args: object) -> None:
            return None

    calls = []

    def factory() -> str:
        calls.append(1)
        return "built-by-factory"

    singleton = LazySingleton(factory)
    singleton._lock = SettingLock(singleton)  # type: ignore[assignment]  # pyright: ignore[reportAttributeAccessIssue]
    assert singleton.get() == "built-by-another-caller"
    assert calls == []


def test_make_default_registry_singleton_builds_and_populates_once() -> None:
    calls = []

    def register(registry: dict) -> None:
        calls.append(1)
        registry["a"] = 1

    singleton = make_default_registry_singleton(dict, register)
    assert calls == []
    assert singleton.get() == {"a": 1}
    assert singleton.get() is singleton.get()
    assert calls == [1]


def test_make_default_registry_singleton_concurrent_get_builds_once() -> None:
    calls = []

    def register(_registry: dict) -> None:
        calls.append(1)

    singleton = make_default_registry_singleton(dict, register)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: singleton.get(), range(32)))
    assert calls == [1]
    assert all(r is results[0] for r in results)


################################################
#     Tests for load_registry_plugins          #
################################################


class _FakeEntryPoint:
    def __init__(self, name: str, func: Callable[[Any], None]) -> None:
        self.name = name
        self._func = func

    def load(self) -> Callable[[Any], None]:
        return self._func


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEntryPoint]) -> None:
    monkeypatch.setattr(singleton, "entry_points", lambda group: eps)  # noqa: ARG005


def _broken(registry: Any) -> NoReturn:  # noqa: ARG001
    msg = "boom"
    raise RuntimeError(msg)


def test_load_registry_plugins_calls_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", lambda r: r.update(a=1))])
    registry: dict[str, int] = {}
    load_registry_plugins(registry, "grp")
    assert registry == {"a": 1}


def test_load_registry_plugins_warns_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("bad", _broken), _FakeEntryPoint("ok", lambda r: r.update(b=2))],
    )
    registry: dict[str, int] = {}
    with pytest.warns(RuntimeWarning, match="Skipping plugin 'bad'"):
        load_registry_plugins(registry, "grp")
    assert registry == {"b": 2}


def test_make_default_registry_singleton_loads_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", lambda r: r.update(p=1))])
    holder = make_default_registry_singleton(dict, lambda r: r.update(d=0), plugin_group="grp")
    assert holder.get() == {"d": 0, "p": 1}

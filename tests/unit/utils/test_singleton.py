from __future__ import annotations

import contextlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Never, cast

import pytest

from coola.utils import fork, singleton
from coola.utils.singleton import (
    DISABLE_PLUGINS_ENV_VAR,
    STRICT_PLUGINS_ENV_VAR,
    LazySingleton,
    LoadedPlugin,
    get_loaded_plugins,
    load_registry_plugins,
    make_default_registry_singleton,
    reset_default_registries,
    temporary_default_registries,
)

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

###################################
#     Tests for LazySingleton     #
###################################


def test_lazy_singleton_get_returns_value_from_factory() -> None:
    singleton = LazySingleton(lambda: 42)
    assert singleton.get() == 42


def test_lazy_singleton_get_returns_none_from_factory() -> None:
    assert LazySingleton(lambda: None).get() is None


def test_lazy_singleton_calls_factory_only_once_when_it_returns_none() -> None:
    calls = []
    singleton_none = LazySingleton(lambda: calls.append(1))
    singleton_none.get()
    singleton_none.get()
    assert len(calls) == 1


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


def test_lazy_singleton_get_in_factory_raises_instead_of_blocking() -> None:
    holder: dict[str, LazySingleton[int]] = {}

    def factory() -> int:
        return holder["singleton"].get()

    holder["singleton"] = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="requested while it is being built"):
        holder["singleton"].get()


def test_lazy_singleton_reentrant_call_raises_error() -> None:
    holder: dict[str, LazySingleton[int]] = {}

    def factory() -> int:
        return holder["singleton"].get()

    holder["singleton"] = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="requested while it is being built"):
        holder["singleton"].get()


def test_lazy_singleton_can_be_built_after_a_reentrant_call_failed() -> None:
    holder: dict[str, LazySingleton[int]] = {}
    calls = []

    def factory() -> int:
        calls.append(1)
        if len(calls) == 1:
            return holder["singleton"].get()
        return 42

    holder["singleton"] = LazySingleton(factory)
    with contextlib.suppress(RuntimeError):
        holder["singleton"].get()
    assert holder["singleton"].get() == 42


def test_lazy_singleton_returns_same_instance() -> None:
    singleton = LazySingleton(object)
    assert singleton.get() is singleton.get()


def test_lazy_singleton_returns_the_same_instance() -> None:
    singleton = LazySingleton(list)
    instance1 = singleton.get()
    instance1.append(1)
    assert instance1 is singleton.get()


def test_lazy_singleton_returns_mutated_instance() -> None:
    singleton = LazySingleton(list)
    singleton.get().append(1)
    assert singleton.get() == [1]


def test_lazy_singleton_mutated_instance_has_the_mutation() -> None:
    singleton1 = LazySingleton(list)
    singleton1.get().append(1)
    assert singleton1.get() == [1]


def test_lazy_singleton_independent_instances_have_independent_state() -> None:
    singleton1 = LazySingleton(list)
    singleton2 = LazySingleton(list)
    singleton1.get().append(1)
    assert singleton2.get() == []


def test_lazy_singleton_falsy_value_is_returned() -> None:
    singleton = LazySingleton(lambda: 0)
    assert singleton.get() == 0


def test_lazy_singleton_falsy_value_is_returned_again() -> None:
    singleton = LazySingleton(lambda: 0)
    singleton.get()
    assert singleton.get() == 0


def test_lazy_singleton_falsy_value_is_cached_correctly() -> None:
    calls = []

    def factory() -> int:
        calls.append(1)
        return 0

    singleton = LazySingleton(factory)
    singleton.get()
    singleton.get()
    assert len(calls) == 1


def test_lazy_singleton_propagates_factory_exception() -> None:
    def factory() -> int:
        msg = "boom"
        raise RuntimeError(msg)

    singleton = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="boom"):
        singleton.get()


def test_lazy_singleton_factory_exception_is_raised() -> None:
    def factory() -> int:
        msg = "boom"
        raise RuntimeError(msg)

    singleton = LazySingleton(factory)
    with pytest.raises(RuntimeError, match="boom"):
        singleton.get()


def make_failing_once_factory(calls: list[int]) -> Callable[[], int]:
    def factory() -> int:
        calls.append(1)
        if len(calls) == 1:
            msg = "boom"
            raise RuntimeError(msg)
        return 42

    return factory


def test_lazy_singleton_recovers_after_factory_exception() -> None:
    singleton = LazySingleton(make_failing_once_factory([]))
    with contextlib.suppress(RuntimeError):
        singleton.get()
    assert singleton.get() == 42


def test_lazy_singleton_calls_the_factory_again_after_factory_exception() -> None:
    calls: list[int] = []
    singleton = LazySingleton(make_failing_once_factory(calls))
    with contextlib.suppress(RuntimeError):
        singleton.get()
    singleton.get()
    assert len(calls) == 2


class SettingLock:
    def __init__(self, singleton: LazySingleton[str]) -> None:
        self._singleton = singleton

    def __enter__(self) -> None:
        self._singleton._instance = "built-by-another-caller"

    def __exit__(self, *args: object) -> None:
        return None


def test_lazy_singleton_get_skips_factory_when_instance_set_during_lock_acquisition() -> None:
    # Covers the double-checked-locking branch: if the instance was built by
    # another caller while this one was waiting to acquire the lock, the
    # factory must not be called again once the lock is held.
    singleton = LazySingleton(lambda: "built-by-factory")
    singleton._lock = cast("threading.Lock", SettingLock(singleton))
    assert singleton.get() == "built-by-another-caller"


def test_lazy_singleton_get_does_not_call_the_factory_when_instance_set_during_lock_acquisition() -> (
    None
):
    calls = []

    def factory() -> str:
        calls.append(1)
        return "built-by-factory"

    singleton = LazySingleton(factory)
    singleton._lock = cast("threading.Lock", SettingLock(singleton))
    singleton.get()
    assert calls == []


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


###########################################
#     Tests for load_registry_plugins     #
###########################################


class _FakeEntryPoint:
    def __init__(self, name: str, func: Callable[[Any], None]) -> None:
        self.name = name
        self.value = f"fake:{name}"
        self._func = func

    def load(self) -> Callable[[Any], None]:
        return self._func


def _patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps: list[_FakeEntryPoint]) -> None:
    monkeypatch.setattr(singleton, "entry_points", lambda group: eps)  # noqa: ARG005


def _broken(registry: Any) -> Never:  # noqa: ARG001
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
    with pytest.warns(RuntimeWarning, match="Skipping plugin 'bad'"):
        load_registry_plugins({}, "grp")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_registry_plugins_loads_the_other_plugins_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("bad", _broken), _FakeEntryPoint("ok", lambda r: r.update(b=2))],
    )
    registry: dict[str, int] = {}
    load_registry_plugins(registry, "grp")
    assert registry == {"b": 2}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_registry_plugins_logs_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bad", _broken)])
    with caplog.at_level("ERROR", logger=singleton.logger.name):
        load_registry_plugins({}, "grp")
    assert "Failed to load plugin 'bad' of group 'grp'" in caplog.text


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_registry_plugins_logs_failure_with_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bad", _broken)])
    with caplog.at_level("ERROR", logger=singleton.logger.name):
        load_registry_plugins({}, "grp")
    assert "RuntimeError: boom" in caplog.text


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_load_registry_plugins_strict_mode_raises(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(STRICT_PLUGINS_ENV_VAR, value)
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bad", _broken)])
    with pytest.RaisesGroup(
        pytest.RaisesExc(
            RuntimeError,
            match="boom",
            check=lambda e: "plugin 'bad' of group 'grp'" in e.__notes__[0],
        ),
        match="Failed to load 1 plugin",
    ):
        load_registry_plugins({}, "grp")


def test_load_registry_plugins_strict_mode_reports_all_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STRICT_PLUGINS_ENV_VAR, "1")
    _patch_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint("bad1", _broken),
            _FakeEntryPoint("ok", lambda r: r.update(p=1)),
            _FakeEntryPoint("bad2", _broken),
        ],
    )
    registry: dict = {}
    with pytest.RaisesGroup(RuntimeError, RuntimeError, match="Failed to load 2 plugin"):
        load_registry_plugins(registry, "grp")
    assert registry == {"p": 1}


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE"])
def test_load_registry_plugins_strict_mode_disabled_warns(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(STRICT_PLUGINS_ENV_VAR, value)
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bad", _broken)])
    with pytest.warns(RuntimeWarning, match="Skipping plugin 'bad'"):
        load_registry_plugins({}, "grp")


def test_make_default_registry_singleton_loads_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", lambda r: r.update(p=1))])
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

    _patch_entry_points(
        monkeypatch, [_FakeEntryPoint("bad", bad_plugin), _FakeEntryPoint("good", good_plugin)]
    )
    holder["singleton"] = make_default_registry_singleton(dict, lambda _: None, "grp")
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


def test_lazy_singleton_is_built_false_initially() -> None:
    assert not LazySingleton(lambda: 42).is_built()


def test_lazy_singleton_is_built_true_after_get() -> None:
    singleton = LazySingleton(lambda: None)
    singleton.get()
    assert singleton.is_built()


def test_lazy_singleton_reset_rebuilds_the_value() -> None:
    calls = []
    singleton = LazySingleton(lambda: calls.append(1) or len(calls))
    singleton.get()
    singleton.reset()
    assert singleton.get() == 2


def test_lazy_singleton_reset_from_the_factory_raises() -> None:
    singleton = LazySingleton(lambda: singleton.reset())
    with pytest.raises(RuntimeError, match="reset while its value is being built"):
        singleton.get()


def test_lazy_singleton_reset_from_the_factory_does_not_build_the_value() -> None:
    singleton = LazySingleton(lambda: singleton.reset())
    with contextlib.suppress(RuntimeError):
        singleton.get()
    assert not singleton.is_built()


def test_lazy_singleton_is_tracked_for_forks() -> None:
    singleton = LazySingleton(lambda: 1)
    assert singleton in fork._TRACKED


def test_lazy_singleton_reset_after_fork_recreates_the_lock() -> None:
    singleton = LazySingleton(lambda: 1)
    lock = singleton._lock
    singleton._reset_after_fork()
    assert singleton._lock is not lock


def test_lazy_singleton_reset_after_fork_does_not_block_if_the_lock_was_held() -> None:
    singleton = LazySingleton(lambda: 1)
    singleton._lock.acquire()  # held by a thread that does not exist in the child
    singleton._reset_after_fork()
    assert singleton.get() == 1


def test_lazy_singleton_reset_after_fork_forgets_the_building_thread() -> None:
    singleton = LazySingleton(lambda: 1)
    singleton._building_thread = 1
    singleton._reset_after_fork()
    assert singleton._building_thread is None


def test_lazy_singleton_reset_after_fork_keeps_the_value() -> None:
    singleton = LazySingleton(object)
    value = singleton.get()
    singleton._reset_after_fork()
    assert singleton.get() is value


#####################################################
#     Tests for the scope of the default registries #
#####################################################


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


#############################################
#     Tests for the disabled / listed plugins
#############################################


def _noop(registry: Any) -> None:
    pass


@pytest.fixture(autouse=True)
def _isolate_loaded_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(singleton, "_LOADED_PLUGINS", {})


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_load_registry_plugins_disabled_does_not_call_the_plugins(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(DISABLE_PLUGINS_ENV_VAR, value)
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", lambda r: r.update(a=1))])
    registry: dict[str, int] = {}
    load_registry_plugins(registry, "grp")
    assert registry == {}


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE"])
def test_load_registry_plugins_disable_variable_unset_calls_the_plugins(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(DISABLE_PLUGINS_ENV_VAR, value)
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", lambda r: r.update(a=1))])
    registry: dict[str, int] = {}
    load_registry_plugins(registry, "grp")
    assert registry == {"a": 1}


def test_load_registry_plugins_disabled_does_not_read_the_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(group: str) -> Never:
        msg = f"entry_points({group!r}) must not be called"
        raise AssertionError(msg)

    monkeypatch.setenv(DISABLE_PLUGINS_ENV_VAR, "1")
    monkeypatch.setattr(singleton, "entry_points", fail)
    load_registry_plugins({}, "grp")


def test_load_registry_plugins_disabled_skips_the_strict_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DISABLE_PLUGINS_ENV_VAR, "1")
    monkeypatch.setenv(STRICT_PLUGINS_ENV_VAR, "1")
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bad", _broken)])
    load_registry_plugins({}, "grp")


def test_get_loaded_plugins_empty() -> None:
    assert get_loaded_plugins() == []


def test_get_loaded_plugins_lists_the_loaded_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", lambda r: r.update(a=1))])
    load_registry_plugins({}, "grp")
    assert get_loaded_plugins() == [LoadedPlugin(group="grp", name="a", value="fake:a")]


def test_get_loaded_plugins_keeps_the_loading_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(
        monkeypatch,
        [_FakeEntryPoint("b", _noop), _FakeEntryPoint("a", _noop)],
    )
    load_registry_plugins({}, "grp")
    assert [plugin.name for plugin in get_loaded_plugins()] == ["b", "a"]


def test_get_loaded_plugins_does_not_list_the_failed_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(
        monkeypatch, [_FakeEntryPoint("bad", _broken), _FakeEntryPoint("good", _noop)]
    )
    with pytest.warns(RuntimeWarning, match="Skipping plugin 'bad'"):
        load_registry_plugins({}, "grp")
    assert [plugin.name for plugin in get_loaded_plugins()] == ["good"]


def test_get_loaded_plugins_lists_a_plugin_loaded_twice_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", _noop)])
    load_registry_plugins({}, "grp")
    load_registry_plugins({}, "grp")
    assert len(get_loaded_plugins()) == 1


def test_get_loaded_plugins_lists_the_same_name_in_two_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", _noop)])
    load_registry_plugins({}, "grp1")
    load_registry_plugins({}, "grp2")
    assert [plugin.group for plugin in get_loaded_plugins()] == ["grp1", "grp2"]


def test_get_loaded_plugins_returns_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", _noop)])
    load_registry_plugins({}, "grp")
    get_loaded_plugins().clear()
    assert len(get_loaded_plugins()) == 1


def test_get_loaded_plugins_after_a_default_registry_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("a", lambda r: r.update(p=1))])
    make_default_registry_singleton(dict, _noop, plugin_group="grp").get()
    assert [plugin.name for plugin in get_loaded_plugins()] == ["a"]

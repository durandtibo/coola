from __future__ import annotations

from typing import Any, Never

import pytest

from coola.registry import plugins
from coola.registry.plugins import (
    DISABLE_PLUGINS_ENV_VAR,
    STRICT_PLUGINS_ENV_VAR,
    LoadedPlugin,
    get_loaded_plugins,
    load_registry_plugins,
)
from tests.unit.registry.helpers import FakeEntryPoint, noop, patch_entry_points

#################################
#     Tests for the plugins     #
#################################


def _broken(registry: Any) -> Never:  # noqa: ARG001
    msg = "boom"
    raise RuntimeError(msg)


def test_load_registry_plugins_calls_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", lambda r: r.update(a=1))])
    registry: dict[str, int] = {}
    load_registry_plugins(registry, "grp")
    assert registry == {"a": 1}


def test_load_registry_plugins_warns_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(
        monkeypatch,
        [FakeEntryPoint("bad", _broken), FakeEntryPoint("ok", lambda r: r.update(b=2))],
    )
    with pytest.warns(RuntimeWarning, match="Skipping plugin 'bad'"):
        load_registry_plugins({}, "grp")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_registry_plugins_loads_the_other_plugins_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_entry_points(
        monkeypatch,
        [FakeEntryPoint("bad", _broken), FakeEntryPoint("ok", lambda r: r.update(b=2))],
    )
    registry: dict[str, int] = {}
    load_registry_plugins(registry, "grp")
    assert registry == {"b": 2}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_registry_plugins_logs_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("bad", _broken)])
    with caplog.at_level("ERROR", logger=plugins.logger.name):
        load_registry_plugins({}, "grp")
    assert "Failed to load plugin 'bad' of group 'grp'" in caplog.text


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_load_registry_plugins_logs_failure_with_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("bad", _broken)])
    with caplog.at_level("ERROR", logger=plugins.logger.name):
        load_registry_plugins({}, "grp")
    assert "RuntimeError: boom" in caplog.text


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_load_registry_plugins_strict_mode_raises(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(STRICT_PLUGINS_ENV_VAR, value)
    patch_entry_points(monkeypatch, [FakeEntryPoint("bad", _broken)])
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
    patch_entry_points(
        monkeypatch,
        [
            FakeEntryPoint("bad1", _broken),
            FakeEntryPoint("ok", lambda r: r.update(p=1)),
            FakeEntryPoint("bad2", _broken),
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
    patch_entry_points(monkeypatch, [FakeEntryPoint("bad", _broken)])
    with pytest.warns(RuntimeWarning, match="Skipping plugin 'bad'"):
        load_registry_plugins({}, "grp")


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_load_registry_plugins_disabled_does_not_call_the_plugins(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(DISABLE_PLUGINS_ENV_VAR, value)
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", lambda r: r.update(a=1))])
    registry: dict[str, int] = {}
    load_registry_plugins(registry, "grp")
    assert registry == {}


@pytest.mark.parametrize("value", ["", "0", "false", "FALSE"])
def test_load_registry_plugins_disable_variable_unset_calls_the_plugins(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(DISABLE_PLUGINS_ENV_VAR, value)
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", lambda r: r.update(a=1))])
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
    monkeypatch.setattr(plugins, "entry_points", fail)
    load_registry_plugins({}, "grp")


def test_load_registry_plugins_disabled_skips_the_strict_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DISABLE_PLUGINS_ENV_VAR, "1")
    monkeypatch.setenv(STRICT_PLUGINS_ENV_VAR, "1")
    patch_entry_points(monkeypatch, [FakeEntryPoint("bad", _broken)])
    load_registry_plugins({}, "grp")


def test_get_loaded_plugins_empty() -> None:
    assert get_loaded_plugins() == []


def test_get_loaded_plugins_lists_the_loaded_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", lambda r: r.update(a=1))])
    load_registry_plugins({}, "grp")
    assert get_loaded_plugins() == [LoadedPlugin(group="grp", name="a", value="fake:a")]


def test_get_loaded_plugins_keeps_the_loading_order(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(
        monkeypatch,
        [FakeEntryPoint("b", noop), FakeEntryPoint("a", noop)],
    )
    load_registry_plugins({}, "grp")
    assert [plugin.name for plugin in get_loaded_plugins()] == ["b", "a"]


def test_get_loaded_plugins_does_not_list_the_failed_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("bad", _broken), FakeEntryPoint("good", noop)])
    with pytest.warns(RuntimeWarning, match="Skipping plugin 'bad'"):
        load_registry_plugins({}, "grp")
    assert [plugin.name for plugin in get_loaded_plugins()] == ["good"]


def test_get_loaded_plugins_lists_a_plugin_loaded_twice_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", noop)])
    load_registry_plugins({}, "grp")
    load_registry_plugins({}, "grp")
    assert len(get_loaded_plugins()) == 1


def test_get_loaded_plugins_lists_the_same_name_in_two_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", noop)])
    load_registry_plugins({}, "grp1")
    load_registry_plugins({}, "grp2")
    assert [plugin.group for plugin in get_loaded_plugins()] == ["grp1", "grp2"]


def test_get_loaded_plugins_returns_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_entry_points(monkeypatch, [FakeEntryPoint("a", noop)])
    load_registry_plugins({}, "grp")
    get_loaded_plugins().clear()
    assert len(get_loaded_plugins()) == 1

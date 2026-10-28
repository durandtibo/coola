from __future__ import annotations

from typing import TYPE_CHECKING, Any

from coola.registry import plugins

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


class FakeEntryPoint:
    def __init__(self, name: str, func: Callable[[Any], None]) -> None:
        self.name = name
        self.value = f"fake:{name}"
        self._func = func

    def load(self) -> Callable[[Any], None]:
        return self._func


def patch_entry_points(monkeypatch: pytest.MonkeyPatch, eps: list[FakeEntryPoint]) -> None:
    monkeypatch.setattr(plugins, "entry_points", lambda group: eps)  # noqa: ARG005


def noop(registry: Any) -> None:
    pass

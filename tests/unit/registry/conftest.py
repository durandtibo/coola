from __future__ import annotations

import pytest

from coola.registry import plugins


@pytest.fixture(autouse=True)
def _isolate_loaded_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugins, "_LOADED_PLUGINS", {})

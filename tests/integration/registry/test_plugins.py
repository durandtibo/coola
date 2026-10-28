from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.integration.helpers import run_python

if TYPE_CHECKING:
    from pathlib import Path

PLUGIN_MODULE = """
from coola.equality.tester import DefaultEqualityTester


class Marker:
    pass


def register(registry):
    registry.register(Marker, DefaultEqualityTester())
"""


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    r"""Create an installed package (a ``dist-info`` directory) with a
    plugin of the equality testers."""
    tmp_path.joinpath("coola_fake_plugin.py").write_text(PLUGIN_MODULE)
    dist_info = tmp_path.joinpath("coola_fake_plugin-1.0.dist-info")
    dist_info.mkdir()
    dist_info.joinpath("METADATA").write_text("Metadata-Version: 2.1\nName: coola-fake-plugin\n")
    dist_info.joinpath("entry_points.txt").write_text(
        "[coola.equality.testers]\nfake = coola_fake_plugin:register\n"
    )
    return tmp_path


def run_with_plugin(plugin_dir: Path, code: str, *, disabled: bool = False) -> list[str]:
    r"""Run code in a fresh interpreter that can find the plugin."""
    setup = f"import os, sys; sys.path.insert(0, {str(plugin_dir)!r}); "
    if disabled:
        setup += "os.environ['COOLA_DISABLE_PLUGINS'] = '1'; "
    return run_python(setup + code)


##########################################
#     Tests for the entry-point plugins   #
##########################################


def test_plugin_is_applied_to_the_default_registry(plugin_dir: Path) -> None:
    """Test that the plugin of an installed package registers its
    type."""
    code = (
        "from coola.equality.tester import get_default_registry; "
        "from coola_fake_plugin import Marker; "
        "print(get_default_registry().has_equality_tester(Marker))"
    )
    assert run_with_plugin(plugin_dir, code) == ["True"]


def test_plugin_is_listed_after_the_registry_is_built(plugin_dir: Path) -> None:
    code = (
        "from coola.equality.tester import get_default_registry; "
        "from coola.registry.plugins import get_loaded_plugins; "
        "get_default_registry(); "
        "print(*[(p.group, p.name, p.value) for p in get_loaded_plugins()])"
    )
    assert run_with_plugin(plugin_dir, code) == [
        "('coola.equality.testers',",
        "'fake',",
        "'coola_fake_plugin:register')",
    ]


def test_plugin_is_not_listed_before_the_registry_is_built(plugin_dir: Path) -> None:
    code = "from coola.registry.plugins import get_loaded_plugins; print(len(get_loaded_plugins()))"
    assert run_with_plugin(plugin_dir, code) == ["0"]


def test_disabled_plugin_is_not_applied(plugin_dir: Path) -> None:
    """Test that ``COOLA_DISABLE_PLUGINS`` prevents the loading of the
    plugins."""
    code = (
        "from coola.equality.tester import get_default_registry; "
        "from coola_fake_plugin import Marker; "
        "print(get_default_registry().has_equality_tester(Marker))"
    )
    assert run_with_plugin(plugin_dir, code, disabled=True) == ["False"]


def test_disabled_plugin_is_not_listed(plugin_dir: Path) -> None:
    code = (
        "from coola.equality.tester import get_default_registry; "
        "from coola.registry.plugins import get_loaded_plugins; "
        "get_default_registry(); "
        "print(len(get_loaded_plugins()))"
    )
    assert run_with_plugin(plugin_dir, code, disabled=True) == ["0"]

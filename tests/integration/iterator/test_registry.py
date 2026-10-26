from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.iterator import (
    BaseChildFinder,
    bfs_iterate,
    dfs_iterate,
    get_default_registry,
)
from coola.testing.fixtures import numpy_available, pandas_available
from coola.utils.imports import is_numpy_available, is_pandas_available
from tests.integration.helpers import run_python

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

if is_numpy_available():
    import numpy as np
if is_pandas_available():
    import pandas as pd


class Pair:
    def __init__(self, left: Any, right: Any) -> None:
        self.left = left
        self.right = right


class PairChildFinder(BaseChildFinder[Pair]):
    def find_children(self, data: Pair) -> list:
        return [data.left, data.right]


PLUGIN_MODULE = """
from coola.iterator import BaseChildFinder


class Marker:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class MarkerChildFinder(BaseChildFinder):
    def find_children(self, data):
        return [data.a, data.b]


def register(registry):
    registry.register(Marker, MarkerChildFinder())
"""


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    r"""Create an installed package (a ``dist-info`` directory) with a
    plugin of the child finders."""
    tmp_path.joinpath("coola_iterator_plugin.py").write_text(PLUGIN_MODULE)
    dist_info = tmp_path.joinpath("coola_iterator_plugin-1.0.dist-info")
    dist_info.mkdir()
    dist_info.joinpath("METADATA").write_text(
        "Metadata-Version: 2.1\nName: coola-iterator-plugin\n"
    )
    dist_info.joinpath("entry_points.txt").write_text(
        "[coola.iterator.child_finders]\nfake = coola_iterator_plugin:register\n"
    )
    return tmp_path


def run_with_plugin(plugin_dir: Path, code: str) -> list[str]:
    r"""Run code in a fresh interpreter that can find the plugin."""
    return run_python(f"import sys; sys.path.insert(0, {str(plugin_dir)!r}); " + code)


@pytest.fixture
def _pair_registered() -> Generator[None, None, None]:
    registry = get_default_registry()
    registry.register(Pair, PairChildFinder())
    yield
    registry.unregister(Pair)


@pytest.mark.usefixtures("_pair_registered")
def test_custom_child_finder_serves_dfs_iterate() -> None:
    assert list(dfs_iterate(Pair(1, [Pair(2, 3), 4]))) == [1, 2, 3, 4]


@pytest.mark.usefixtures("_pair_registered")
def test_custom_child_finder_serves_bfs_iterate() -> None:
    assert list(bfs_iterate(Pair(1, [Pair(2, 3), 4]))) == [1, 4, 2, 3]


def test_plugin_child_finder_serves_both_orders(plugin_dir: Path) -> None:
    code = (
        "from coola.iterator import bfs_iterate, dfs_iterate; "
        "from coola_iterator_plugin import Marker; "
        "data = Marker(1, [Marker(2, 3), 4]); "
        "print(list(dfs_iterate(data)), list(bfs_iterate(data)))"
    )
    assert run_with_plugin(plugin_dir, code) == ["[1,", "2,", "3,", "4]", "[1,", "4,", "2,", "3]"]


@numpy_available
def test_numpy_array_is_a_leaf() -> None:
    array = np.arange(3)
    assert next(dfs_iterate([array, 1])) is array


@pandas_available
def test_pandas_dataframe_is_a_leaf() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    assert list(bfs_iterate([frame])) == [frame]


def test_import_does_not_import_the_optional_libraries() -> None:
    code = (
        "import sys; import coola.iterator; "
        "print(any(name in sys.modules for name in ('numpy', 'pandas', 'torch')))"
    )
    assert run_python(code) == ["False"]

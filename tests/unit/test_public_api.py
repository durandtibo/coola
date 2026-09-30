from __future__ import annotations

import importlib
import pkgutil

import pytest

import coola


def _public_modules() -> list[str]:
    names = [coola.__name__]
    for info in pkgutil.walk_packages(coola.__path__, prefix="coola."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("name", _public_modules())
def test_public_module_has_all(name: str) -> None:
    module = importlib.import_module(name)
    assert hasattr(module, "__all__"), f"{name} has no __all__"


@pytest.mark.parametrize("name", _public_modules())
def test_all_names_are_importable(name: str) -> None:
    module = importlib.import_module(name)
    for attr in getattr(module, "__all__", []):
        assert hasattr(module, attr), f"{name}.__all__ lists missing name {attr!r}"


@pytest.mark.parametrize(
    ("module", "alias"),
    [
        ("coola.equality.tester", "get_default_equality_tester_registry"),
        ("coola.hashing", "get_default_hasher_registry"),
        ("coola.iterator.bfs", "get_default_child_finder_registry"),
        ("coola.iterator.dfs", "get_default_iterator_registry"),
        ("coola.random", "get_default_random_manager_registry"),
        ("coola.recursive", "get_default_transformer_registry"),
        ("coola.summary", "get_default_summarizer_registry"),
    ],
)
def test_default_registry_alias(module: str, alias: str) -> None:
    mod = importlib.import_module(module)
    assert getattr(mod, alias) is mod.get_default_registry

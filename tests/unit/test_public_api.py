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


_PUBLIC_NAMES = [
    (name, attr)
    for name in _public_modules()
    for attr in getattr(importlib.import_module(name), "__all__", [])
]


@pytest.mark.parametrize(("name", "attr"), _PUBLIC_NAMES)
def test_all_names_are_importable(name: str, attr: str) -> None:
    module = importlib.import_module(name)
    assert hasattr(module, attr), f"{name}.__all__ lists missing name {attr!r}"


@pytest.mark.parametrize(
    ("module", "alias"),
    [
        ("coola.hashing", "get_default_hasher_registry"),
        ("coola.iterator", "get_default_child_finder_registry"),
        ("coola.random", "get_default_random_manager_registry"),
        ("coola.recursive", "get_default_transformer_registry"),
        ("coola.summary", "get_default_summarizer_registry"),
    ],
)
def test_default_registry_alias(module: str, alias: str) -> None:
    mod = importlib.import_module(module)
    assert getattr(mod, alias) is mod.get_default_registry

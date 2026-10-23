from __future__ import annotations

import importlib

import pytest

from coola.utils.singleton import LazySingleton

REGISTRIES = [
    ("coola.equality.tester.interface", "EqualityTesterRegistry"),
    ("coola.hashing.interface", "HasherRegistry"),
    ("coola.iterator.bfs.interface", "ChildFinderRegistry"),
    ("coola.iterator.dfs.interface", "IteratorRegistry"),
    ("coola.random.interface", "RandomManagerRegistry"),
    ("coola.recursive.interface", "TransformerRegistry"),
    ("coola.summary.interface", "SummarizerRegistry"),
]


@pytest.mark.parametrize("module", [module for module, _ in REGISTRIES])
def test_default_registry_is_lazy_singleton(module: str) -> None:
    assert isinstance(importlib.import_module(module)._default_registry, LazySingleton)


@pytest.mark.parametrize(("module", "registry_name"), REGISTRIES)
def test_default_registry_is_lazy_singleton_of_expected_type(
    module: str, registry_name: str
) -> None:
    assert type(importlib.import_module(module).get_default_registry()).__name__ == registry_name


@pytest.mark.parametrize("module", [module for module, _ in REGISTRIES])
def test_default_registry_is_returned_every_time(module: str) -> None:
    mod = importlib.import_module(module)
    assert mod.get_default_registry() is mod.get_default_registry()

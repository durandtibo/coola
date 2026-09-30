from __future__ import annotations

import importlib

import pytest

from coola.utils.singleton import LazySingleton


@pytest.mark.parametrize(
    ("module", "registry_name"),
    [
        ("coola.equality.tester.interface", "EqualityTesterRegistry"),
        ("coola.hashing.interface", "HasherRegistry"),
        ("coola.iterator.bfs.interface", "ChildFinderRegistry"),
        ("coola.iterator.dfs.interface", "IteratorRegistry"),
        ("coola.random.interface", "RandomManagerRegistry"),
        ("coola.recursive.interface", "TransformerRegistry"),
        ("coola.summary.interface", "SummarizerRegistry"),
    ],
)
def test_default_registry_is_lazy_singleton_of_expected_type(
    module: str, registry_name: str
) -> None:
    mod = importlib.import_module(module)
    assert isinstance(mod._default_registry, LazySingleton)
    registry = mod.get_default_registry()
    assert type(registry).__name__ == registry_name
    assert mod.get_default_registry() is registry

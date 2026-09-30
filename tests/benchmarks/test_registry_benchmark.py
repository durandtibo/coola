r"""Benchmarks for type registry resolution.

Run them explicitly with:

    inv benchmark
"""

from __future__ import annotations

from coola.registry import TypeRegistry


def test_type_registry_resolve_cached_benchmark(benchmark) -> None:  # noqa: ANN001
    registry = TypeRegistry[str]({object: "object", int: "int"})
    registry.resolve(bool)  # warm the cache
    assert benchmark(registry.resolve, bool) == "int"


def test_type_registry_resolve_uncached_benchmark(benchmark) -> None:  # noqa: ANN001
    registry = TypeRegistry[str]({object: "object", int: "int"})

    def resolve() -> str:
        registry._cache = {}
        return registry.resolve(bool)

    assert benchmark(resolve) == "int"

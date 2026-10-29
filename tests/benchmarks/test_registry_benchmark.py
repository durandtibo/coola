r"""Benchmarks for the registries.

Run them explicitly with:

    inv test.benchmark

They exist to compare the performance of the implementations of
``coola.registry`` over time (``review/registry.md``). They only use the
public methods that exist before and after the redesign (``resolve``,
``has``, ``register``, ``copy``, ``items``, ...) and the default registries
of the packages built on it, so that they run unchanged on any design of the
package. The only exception is ``test_type_registry_resolve_uncached_benchmark``,
which resets the private cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from coola.equality.tester import get_default_registry as get_tester_registry
from coola.hashing import get_default_registry as get_hasher_registry, hash_object
from coola.recursive import get_default_registry as get_transformer_registry
from coola.registry import Registry, TypeRegistry

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


@dataclass
class Point:
    x: int = 1
    y: int = 2


class A:
    pass


class B(A):
    pass


class C(B):
    pass


class D(C):
    pass


def make_type_registry(size: int = 50) -> TypeRegistry[int]:
    r"""Create a registry of ``size`` distinct types (``object`` and
    ``size - 1`` classes)."""
    registry = TypeRegistry[int]({object: 0})
    for i in range(size - 1):
        registry.register(type(f"T{i}", (), {}), i)
    return registry


def make_registry(size: int = 50) -> Registry[str, int]:
    return Registry[str, int]({f"key{i}": i for i in range(size)})


########################################
#     Tests for TypeRegistry.resolve   #
########################################


def test_type_registry_resolve_cached_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = TypeRegistry[str]({object: "object", int: "int"})
    registry.resolve(bool)  # warm the cache
    assert benchmark(registry.resolve, bool) == "int"


def test_type_registry_resolve_uncached_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = TypeRegistry[str]({object: "object", int: "int"})

    def resolve() -> str:
        registry._cache = {}
        return registry.resolve(bool)

    assert benchmark(resolve) == "int"


def test_type_registry_resolve_deep_mro_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = TypeRegistry[str]({object: "object", A: "a"})
    registry.resolve(D)  # warm the cache
    assert benchmark(registry.resolve, D) == "a"


def test_type_registry_resolve_abc_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = TypeRegistry[str]({object: "object", Mapping: "mapping"})
    registry.resolve(dict)  # warm the cache
    assert benchmark(registry.resolve, dict) == "object"


def test_type_registry_resolve_cold_types_benchmark(benchmark: BenchmarkFixture) -> None:
    r"""Resolve 5,000 distinct types once each, in a new registry."""

    def setup() -> tuple[tuple[TypeRegistry[int], list[type]], dict[str, Any]]:
        types: list[type] = [type(f"C{i}", (), {}) for i in range(5000)]
        return (TypeRegistry[int]({object: 0}), types), {}

    def resolve_all(registry: TypeRegistry[int], types: list[type]) -> int:
        return sum(registry.resolve(t) for t in types)

    assert benchmark.pedantic(resolve_all, setup=setup, rounds=5) == 0


########################################
#     Tests for the other lookups     #
########################################


def test_type_registry_has_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_type_registry()
    assert benchmark(registry.has, object)


def test_registry_has_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_registry()
    assert benchmark(registry.has, "key10")


def test_registry_get_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_registry()
    assert benchmark(registry.get, "key10") == 10


def test_registry_len_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_registry()
    assert benchmark(len, registry) == 50


def test_registry_items_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_registry()
    assert len(benchmark(lambda: list(registry.items()))) == 50


########################################
#     Tests for the mutations         #
########################################


def test_registry_register_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_registry()
    benchmark(registry.register, "key10", 0, exist_ok=True)
    assert registry.get("key10") == 0


def test_registry_register_many_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_registry()
    mapping = {f"key{i}": 0 for i in range(10)}
    benchmark(registry.register_many, mapping, exist_ok=True)
    assert registry.get("key9") == 0


def test_type_registry_register_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_type_registry()
    benchmark(registry.register, A, 1, exist_ok=True)
    assert registry.resolve(A) == 1


def test_registry_copy_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = make_registry()
    assert len(benchmark(registry.copy)) == 50


########################################
#     Tests for the default registries #
########################################


def test_default_transformer_registry_find_list_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = get_transformer_registry()
    registry.find_transformer(list)  # warm the cache
    benchmark(registry.find_transformer, list)


def test_default_transformer_registry_find_dataclass_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    registry = get_transformer_registry()
    registry.find_transformer(Point)  # warm the cache
    benchmark(registry.find_transformer, Point)


def test_default_hasher_registry_find_dataclass_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = get_hasher_registry()
    registry.find_hasher(Point)  # warm the cache
    benchmark(registry.find_hasher, Point)


def test_default_tester_registry_find_dataclass_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = get_tester_registry()
    registry.find_equality_tester(Point)  # warm the cache
    benchmark(registry.find_equality_tester, Point)


def test_default_hasher_registry_copy_benchmark(benchmark: BenchmarkFixture) -> None:
    registry = get_hasher_registry()
    assert len(benchmark(registry.copy)) == len(registry)


#########################################################
#     Tests for the engines that look up every node     #
#########################################################


def test_hash_object_flat_list_benchmark(benchmark: BenchmarkFixture) -> None:
    r"""``hash_object`` looks the hasher of every node up in the
    registry."""
    data = list(range(10_000))
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_list_of_dataclasses_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [Point(i, i) for i in range(1_000)]
    assert len(benchmark(hash_object, data)) == 64

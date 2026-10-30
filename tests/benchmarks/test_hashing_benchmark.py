r"""Benchmarks for ``hash_object``.

Run them explicitly with:

    inv test.benchmark

They exist to compare the performance of the implementations of
``coola.hashing`` over time. They only use ``hash_object(data)`` with the
default registry, so that they run unchanged on any design of the package.
Do not use the internals of the package (hashers, registry methods) in this
file.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from coola.hashing import digest, get_secret_hashers, get_stable_registry, hash_object, tagged
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

if is_numpy_available():
    import numpy as np


class Point(NamedTuple):
    x: Any
    y: Any


@dataclass
class Record:
    name: str
    value: float
    tags: list[str]


def make_deep(depth: int) -> Any:
    r"""Create a list nested ``depth`` times around a leaf."""
    data: Any = 0
    for _ in range(depth):
        data = [data]
    return data


def make_tree(depth: int, width: int) -> Any:
    r"""Create a balanced tree of dicts and lists with ``width **
    depth`` leaves."""
    if depth == 0:
        return 1
    children = [make_tree(depth - 1, width) for _ in range(width)]
    return children if depth % 2 else {f"k{i}": c for i, c in enumerate(children)}


def make_mixed(size: int) -> Any:
    r"""Create data with many built-in container types."""
    return {
        "list": list(range(size)),
        "tuple": tuple(range(size)),
        "deque": deque(range(size)),
        "set": set(range(size)),
        "frozenset": frozenset(range(size)),
        "ordered": OrderedDict((i, i) for i in range(size)),
        "named_tuple": [Point(i, i) for i in range(size)],
        "strings": [str(i) for i in range(size)],
    }


###########################
#     Flat containers     #
###########################


def test_hash_object_flat_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = list(range(100_000))
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_flat_tuple_benchmark(benchmark: BenchmarkFixture) -> None:
    data = tuple(range(100_000))
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_flat_set_benchmark(benchmark: BenchmarkFixture) -> None:
    data = set(range(100_000))
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_flat_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    data = {i: i for i in range(100_000)}
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_flat_list_of_strings_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [str(i) for i in range(100_000)]
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_flat_list_of_floats_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [i / 7 for i in range(100_000)]
    assert len(benchmark(hash_object, data)) == 64


###########################
#     Small inputs        #
###########################


def test_hash_object_scalar_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(hash_object, 1)) == 64


def test_hash_object_empty_list_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(hash_object, [])) == 64


def test_hash_object_small_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    data = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_dataclass_benchmark(benchmark: BenchmarkFixture) -> None:
    data = Record("a", 1.5, ["x", "y"])
    assert len(benchmark(hash_object, data)) == 64


###########################
#     Nested containers   #
###########################


@pytest.mark.parametrize(("depth", "width"), [(4, 10), (10, 3), (14, 2)])
def test_hash_object_tree_benchmark(benchmark: BenchmarkFixture, depth: int, width: int) -> None:
    data = make_tree(depth, width)
    assert len(benchmark(hash_object, data)) == 64


@pytest.mark.parametrize("depth", [10, 100, 300])
def test_hash_object_deep_list_benchmark(benchmark: BenchmarkFixture, depth: int) -> None:
    data = make_deep(depth)
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_list_of_small_dicts_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [{"a": i, "b": [i], "c": (i,)} for i in range(10_000)]
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_list_of_dataclasses_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [Record(str(i), i / 3, ["x", "y"]) for i in range(10_000)]
    assert len(benchmark(hash_object, data)) == 64


def test_hash_object_mixed_containers_benchmark(benchmark: BenchmarkFixture) -> None:
    data = make_mixed(10_000)
    assert len(benchmark(hash_object, data)) == 64


###########################
#     Arrays and bytes    #
###########################


def test_hash_object_bytes_benchmark(benchmark: BenchmarkFixture) -> None:
    data = bytes(10_000_000)
    assert len(benchmark(hash_object, data)) == 64


@numpy_available
def test_hash_object_numpy_array_benchmark(benchmark: BenchmarkFixture) -> None:
    data = np.arange(1_000_000, dtype=np.float64)
    assert len(benchmark(hash_object, data)) == 64


@numpy_available
def test_hash_object_list_of_numpy_arrays_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [np.arange(10, dtype=np.float64) for _ in range(1_000)]
    assert len(benchmark(hash_object, data)) == 64


def test_digest_benchmark(benchmark: BenchmarkFixture) -> None:
    assert len(benchmark(digest, b"x" * 1_000_000)) == 64


def test_tagged_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(tagged, "str", "abc").startswith(b"str")


def test_get_secret_hashers_benchmark(benchmark: BenchmarkFixture) -> None:
    assert isinstance(benchmark(get_secret_hashers, "exclude"), dict)


def test_get_stable_registry_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark(get_stable_registry)

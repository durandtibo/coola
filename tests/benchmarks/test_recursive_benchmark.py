r"""Benchmarks for ``recursive_apply``.

Run them explicitly with:

    inv test.benchmark

They exist to compare the performance of the implementations of
``coola.recursive`` over time. They only use ``recursive_apply(data, func)``
with the default registry, so that they run unchanged on any design of the
package. Do not use the internals of the package (transformers, registry
methods) in this file.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from coola.recursive import recursive_apply

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


class Point(NamedTuple):
    x: Any
    y: Any


def identity(x: Any) -> Any:
    return x


def double(x: Any) -> Any:
    return x * 2


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
    r"""Create data with every built-in container type."""
    return {
        "list": list(range(size)),
        "tuple": tuple(range(size)),
        "deque": deque(range(size)),
        "set": set(range(size)),
        "frozenset": frozenset(range(size)),
        "named_tuple": [Point(i, i) for i in range(size)],
        "strings": [str(i) for i in range(size)],
    }


###########################
#     Flat containers     #
###########################


def test_recursive_apply_flat_list_benchmark(benchmark: BenchmarkFixture) -> None:
    data = list(range(100_000))
    assert benchmark(recursive_apply, data, double)[:3] == [0, 2, 4]


def test_recursive_apply_flat_tuple_benchmark(benchmark: BenchmarkFixture) -> None:
    data = tuple(range(100_000))
    assert benchmark(recursive_apply, data, double)[:3] == (0, 2, 4)


def test_recursive_apply_flat_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    data = {f"k{i}": i for i in range(100_000)}
    assert benchmark(recursive_apply, data, double)["k2"] == 4


def test_recursive_apply_flat_set_benchmark(benchmark: BenchmarkFixture) -> None:
    data = set(range(100_000))
    assert len(benchmark(recursive_apply, data, double)) == 100_000


def test_recursive_apply_flat_list_identity_benchmark(benchmark: BenchmarkFixture) -> None:
    data = list(range(100_000))
    assert benchmark(recursive_apply, data, identity) == data


def test_recursive_apply_flat_list_of_strings_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [str(i) for i in range(100_000)]
    assert benchmark(recursive_apply, data, identity) == data


#######################
#     Nested data     #
#######################


@pytest.mark.parametrize(("depth", "width"), [(4, 10), (10, 3), (14, 2)])
def test_recursive_apply_tree_benchmark(
    benchmark: BenchmarkFixture, depth: int, width: int
) -> None:
    data = make_tree(depth, width)
    assert benchmark(recursive_apply, data, double) is not None


def test_recursive_apply_list_of_small_dicts_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [{"a": i, "b": [i, i + 1], "c": (i, "x")} for i in range(10_000)]
    assert benchmark(recursive_apply, data, identity)[1] == {"a": 1, "b": [1, 2], "c": (1, "x")}


def test_recursive_apply_mixed_containers_benchmark(benchmark: BenchmarkFixture) -> None:
    data = make_mixed(10_000)
    assert benchmark(recursive_apply, data, identity)["named_tuple"][1] == Point(1, 1)


@pytest.mark.parametrize("depth", [10, 100, 300])
def test_recursive_apply_deep_list_benchmark(benchmark: BenchmarkFixture, depth: int) -> None:
    # Deeper than 300 levels is not benchmarked: a recursive design hits the recursion limit.
    data = make_deep(depth)
    assert benchmark(recursive_apply, data, double) is not None


#########################
#     Call overhead     #
#########################


def test_recursive_apply_scalar_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(recursive_apply, 1, double) == 2


def test_recursive_apply_empty_list_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(recursive_apply, [], double) == []


def test_recursive_apply_small_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    data = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    assert benchmark(recursive_apply, data, double) == {"a": 2, "b": [4, 6], "c": {"d": 8}}

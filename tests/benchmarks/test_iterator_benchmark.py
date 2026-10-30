r"""Benchmarks for ``bfs_iterate`` and ``dfs_iterate``.

Run them explicitly with:

    inv test.benchmark

They exist to compare the performance of the implementations of
``coola.iterator`` over time. They only use ``bfs_iterate(data)`` and
``dfs_iterate(data)`` with the default registry, so that they run unchanged on
any design of the package. Do not use the internals of the package (iterators,
child finders, registry methods) in this file.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from coola.iterator import bfs_iterate, dfs_items, dfs_iterate, filter_by_type

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pytest_benchmark.fixture import BenchmarkFixture


class Point(NamedTuple):
    x: Any
    y: Any


def consume(iterate: Callable[[Any], Iterator[Any]], data: Any) -> int:
    r"""Exhaust the iterator and return the number of leaves."""
    return sum(1 for _ in iterate(data))


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


ITERATE = pytest.mark.parametrize("iterate", [bfs_iterate, dfs_iterate], ids=["bfs", "dfs"])

###########################
#     Flat containers     #
###########################


@ITERATE
def test_iterate_flat_list_benchmark(benchmark: BenchmarkFixture, iterate: Callable) -> None:
    data = list(range(100_000))
    assert benchmark(consume, iterate, data) == 100_000


@ITERATE
def test_iterate_flat_tuple_benchmark(benchmark: BenchmarkFixture, iterate: Callable) -> None:
    data = tuple(range(100_000))
    assert benchmark(consume, iterate, data) == 100_000


@ITERATE
def test_iterate_flat_dict_benchmark(benchmark: BenchmarkFixture, iterate: Callable) -> None:
    data = {i: i for i in range(100_000)}
    assert benchmark(consume, iterate, data) == 100_000


@ITERATE
def test_iterate_flat_set_benchmark(benchmark: BenchmarkFixture, iterate: Callable) -> None:
    data = set(range(100_000))
    assert benchmark(consume, iterate, data) == 100_000


@ITERATE
def test_iterate_flat_list_of_strings_benchmark(
    benchmark: BenchmarkFixture, iterate: Callable
) -> None:
    data = [str(i) for i in range(100_000)]
    assert benchmark(consume, iterate, data) == 100_000


#######################
#     Nested data     #
#######################


@ITERATE
@pytest.mark.parametrize(("depth", "width"), [(4, 10), (10, 3), (14, 2)])
def test_iterate_tree_benchmark(
    benchmark: BenchmarkFixture, iterate: Callable, depth: int, width: int
) -> None:
    data = make_tree(depth, width)
    assert benchmark(consume, iterate, data) == width**depth


@ITERATE
def test_iterate_list_of_small_dicts_benchmark(
    benchmark: BenchmarkFixture, iterate: Callable
) -> None:
    data = [{"a": i, "b": [i, i], "c": (i, i)} for i in range(10_000)]
    assert benchmark(consume, iterate, data) == 50_000


@ITERATE
def test_iterate_mixed_containers_benchmark(benchmark: BenchmarkFixture, iterate: Callable) -> None:
    data = make_mixed(10_000)
    assert benchmark(consume, iterate, data) == 80_000


@ITERATE
@pytest.mark.parametrize("depth", [10, 100, 300])
def test_iterate_deep_list_benchmark(
    benchmark: BenchmarkFixture, iterate: Callable, depth: int
) -> None:
    data = make_deep(depth)
    assert benchmark(consume, iterate, data) == 1


#########################
#     Small inputs      #
#########################


@ITERATE
def test_iterate_scalar_benchmark(benchmark: BenchmarkFixture, iterate: Callable) -> None:
    assert benchmark(consume, iterate, 1) == 1


@ITERATE
def test_iterate_empty_list_benchmark(benchmark: BenchmarkFixture, iterate: Callable) -> None:
    assert benchmark(consume, iterate, []) == 0


@ITERATE
def test_iterate_small_dict_benchmark(benchmark: BenchmarkFixture, iterate: Callable) -> None:
    data = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    assert benchmark(consume, iterate, data) == 4


#####################
#     dfs_items     #
#####################


def test_dfs_items_flat_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    data = {i: i for i in range(100_000)}
    assert benchmark(consume, dfs_items, data) == 100_000


@pytest.mark.parametrize(("depth", "width"), [(4, 10), (14, 2)])
def test_dfs_items_tree_benchmark(benchmark: BenchmarkFixture, depth: int, width: int) -> None:
    data = make_tree(depth, width)
    assert benchmark(consume, dfs_items, data) == width**depth


def test_dfs_items_empty_option_benchmark(benchmark: BenchmarkFixture) -> None:
    data = make_tree(4, 10)
    assert benchmark(lambda: sum(1 for _ in dfs_items(data, empty=True))) == 10_000


def test_filter_by_type_benchmark(benchmark: BenchmarkFixture) -> None:
    data = [1, "a", 2.0] * 10_000
    assert benchmark(lambda: sum(1 for _ in filter_by_type(data, int))) == 10_000

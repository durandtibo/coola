r"""Benchmarks for ``summarize``.

Run them explicitly with:

    inv test.benchmark

They exist to compare the performance of the implementations of
``coola.summary`` over time. They only use ``summarize(data, max_depth=...)``
with the default registry, so that they run unchanged on any design of the
package. Do not use the internals of the package (summarizers, registry
methods) in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from coola.summary import summarize

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


@dataclass
class Config:
    name: str
    values: list[int]
    options: dict[str, Any]


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


###########################
#     Small inputs        #
###########################


def test_summarize_scalar_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark(summarize, 1)


def test_summarize_empty_list_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark(summarize, [])


def test_summarize_small_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark(summarize, {"a": 1, "b": [2, 3], "c": {"d": 4}})


def test_summarize_dataclass_benchmark(benchmark: BenchmarkFixture) -> None:
    data = Config("a", [1, 2, 3], {"x": 1})
    benchmark(lambda: summarize(data, max_depth=2))


@pytest.mark.parametrize("max_depth", [1, 3])
def test_summarize_tree_benchmark(benchmark: BenchmarkFixture, max_depth: int) -> None:
    data = {
        "users": [{"name": "A", "scores": [1, 2, 3]}] * 20,
        "meta": {"k": list(range(100))},
        "s": {1, 2, 3},
    }
    benchmark(lambda: summarize(data, max_depth=max_depth))


###########################
#     Large inputs        #
###########################


@pytest.mark.parametrize("max_depth", [1, 2])
def test_summarize_flat_list_benchmark(benchmark: BenchmarkFixture, max_depth: int) -> None:
    data = list(range(100_000))
    benchmark(lambda: summarize(data, max_depth=max_depth))


def test_summarize_flat_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    data = {i: i for i in range(100_000)}
    benchmark(lambda: summarize(data, max_depth=2))


def test_summarize_flat_set_benchmark(benchmark: BenchmarkFixture) -> None:
    data = set(range(100_000))
    benchmark(lambda: summarize(data, max_depth=2))


def test_summarize_big_children_at_the_display_depth_benchmark(
    benchmark: BenchmarkFixture,
) -> None:
    r"""Five big lists shown in the compact form."""
    data = [[list(range(200_000)) for _ in range(5)]]
    benchmark(lambda: summarize(data, max_depth=1))


@pytest.mark.parametrize("depth", [10, 100, 300])
def test_summarize_deep_list_benchmark(benchmark: BenchmarkFixture, depth: int) -> None:
    data = make_deep(depth)
    benchmark(lambda: summarize(data, max_depth=depth + 1))


def test_summarize_tree_expanded_benchmark(benchmark: BenchmarkFixture) -> None:
    data = make_tree(8, 3)
    benchmark(lambda: summarize(data, max_depth=8))

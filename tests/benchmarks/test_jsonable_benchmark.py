r"""Benchmarks for ``coola.jsonable.to_jsonable``.

Run them explicitly with:

    inv test.benchmark

They are the gate of ``review/json.md``: a change is rejected if it is
slower than ``main`` on one of these inputs, except the array, where the
gate is "at most 6x ``main`` and at most 40 % of ``json.dumps`` of the
result". They only use the public function with its default registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from coola.jsonable import to_jsonable
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

if is_numpy_available():
    import numpy as np


@dataclass
class Pair:
    a: int
    b: int


def make_tree(depth: int, width: int) -> Any:
    r"""Create a balanced tree of dicts and lists with ``width **
    depth`` leaves."""
    if depth == 0:
        return 1
    children = [make_tree(depth - 1, width) for _ in range(width)]
    return children if depth % 2 else {f"k{i}": c for i, c in enumerate(children)}


def make_deep_list(depth: int) -> list[Any]:
    r"""Create a list nested ``depth`` times."""
    data: list[Any] = []
    for _ in range(depth):
        data = [data]
    return data


INPUTS = {
    "flat_int": lambda: list(range(100_000)),
    "flat_str": lambda: [str(i) for i in range(100_000)],
    "dict": lambda: {f"key{i}": i for i in range(100_000)},
    "tree": lambda: {"root": make_tree(10, 3)},
    "dataclasses": lambda: [Pair(i, i) for i in range(10_000)],
    "scalar": lambda: 1,
    "small_dict": lambda: {"a": 1, "b": [2, 3]},
    "deep_list": lambda: make_deep_list(100_000),
}


@pytest.mark.parametrize("name", INPUTS)
def test_to_jsonable_benchmark(benchmark: BenchmarkFixture, name: str) -> None:
    data = INPUTS[name]()
    benchmark(to_jsonable, data)


@numpy_available
def test_to_jsonable_array_benchmark(benchmark: BenchmarkFixture) -> None:
    data = np.random.default_rng(0).random(1_000_000)
    assert len(benchmark(to_jsonable, data)) == 1_000_000

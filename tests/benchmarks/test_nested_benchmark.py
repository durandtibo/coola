r"""Benchmarks for the functions of ``coola.nested`` that look inside
the values.

Run them explicitly with:

    inv test.benchmark

They are the gate of ``review/nested.md``: a change is rejected if a
large input is more than 10 % slower than ``main`` or a tiny input costs
more than 15 µs more. They only use the public functions with their
default registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.nested import (
    add_prefix_suffix_to_keys,
    convert_to_jsonable,
    from_flat_dict,
    remove_keys_if,
    to_flat_dict,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture


def make_tree(depth: int, width: int) -> Any:
    r"""Create a balanced tree of dicts and lists with ``width **
    depth`` leaves."""
    if depth == 0:
        return 1
    children = [make_tree(depth - 1, width) for _ in range(width)]
    return children if depth % 2 else {f"k{i}": c for i, c in enumerate(children)}


def is_key_0(key: Any) -> bool:
    return key == "k0"


TREE = make_tree(5, 6)  # 7,776 leaves
FLAT = {f"key{i}": i for i in range(100_000)}
TINY = {"a": 1, "b": {"c": 2}}

FUNCTIONS = {
    "to_flat_dict": to_flat_dict,
    "remove_keys_if": lambda data: remove_keys_if(data, is_key_0),
    "add_prefix_suffix_to_keys": lambda data: add_prefix_suffix_to_keys(data, "x_", recursive=True),
    "convert_to_jsonable": convert_to_jsonable,
}
FUNCTION = pytest.mark.parametrize("name", FUNCTIONS)


@FUNCTION
def test_tree_benchmark(benchmark: BenchmarkFixture, name: str) -> None:
    data = TREE if isinstance(TREE, dict) else {"root": TREE}
    benchmark(FUNCTIONS[name], data)


@FUNCTION
def test_flat_dict_benchmark(benchmark: BenchmarkFixture, name: str) -> None:
    benchmark(FUNCTIONS[name], FLAT)


@FUNCTION
def test_tiny_input_benchmark(benchmark: BenchmarkFixture, name: str) -> None:
    benchmark(FUNCTIONS[name], TINY)


def test_from_flat_dict_tree_benchmark(benchmark: BenchmarkFixture) -> None:
    flat = to_flat_dict({"root": TREE})
    benchmark(from_flat_dict, flat)


def test_from_flat_dict_restore_lists_benchmark(benchmark: BenchmarkFixture) -> None:
    flat = to_flat_dict({"root": TREE})
    benchmark(lambda: from_flat_dict(flat, restore_lists=True))


def test_from_flat_dict_flat_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark(from_flat_dict, FLAT)


def test_from_flat_dict_tiny_input_benchmark(benchmark: BenchmarkFixture) -> None:
    benchmark(from_flat_dict, {"a": 1, "b.c": 2})

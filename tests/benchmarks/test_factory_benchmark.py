r"""Benchmarks for ``coola.factory``.

Run them explicitly with:

    inv test.benchmark

They only use the public functions of ``coola.factory``, so that they
run unchanged on any design of the package.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from coola.factory import import_object, instantiate_object, is_object_config, resolve_object

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

CONFIG = {"_target_": "collections.OrderedDict"}
PATH = "collections.OrderedDict"


def test_import_object_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(import_object, PATH) is OrderedDict


def test_import_object_with_prefixes_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(import_object, PATH, allowed_prefixes=["collections."]) is OrderedDict


def test_instantiate_object_class_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(instantiate_object, dict, a=1, b=2) == {"a": 1, "b": 2}


def test_instantiate_object_ordered_dict_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(instantiate_object, OrderedDict) == OrderedDict()


def test_resolve_object_config_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(resolve_object, CONFIG, dict) == OrderedDict()


def test_resolve_object_instance_benchmark(benchmark: BenchmarkFixture) -> None:
    obj = OrderedDict()
    assert benchmark(resolve_object, obj, dict) is obj


def test_is_object_config_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(is_object_config, CONFIG, dict, import_target=True)


def test_is_object_config_no_import_benchmark(benchmark: BenchmarkFixture) -> None:
    assert benchmark(is_object_config, CONFIG, dict)

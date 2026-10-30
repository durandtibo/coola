r"""Benchmarks for the loaders and savers of ``coola.io``.

Run them explicitly with:

    inv test.benchmark

They exist to compare the performance of the implementations of
``coola.io`` over time. They only use the public functions and classes
(``save_*``, ``load_*``, ``JsonSaver``, ``JsonLoader``), so that they run
unchanged on any design of the package. Do not use the internals of the
package (locks, temporary files) in this file.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from coola.io import (
    JsonLoader,
    JsonSaver,
    load_json,
    load_pickle,
    load_text,
    load_torch,
    resolve_loader,
    resolve_saver,
    save_json,
    save_pickle,
    save_text,
    save_torch,
)
from coola.testing.marks import torch_available
from coola.utils.imports import is_torch_available

if is_torch_available():
    import torch

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture

SMALL = {"key1": [1, 2, 3], "key2": "abc"}
# About 590 KB of JSON.
LARGE_JSON = list(range(100_000))
ONE_MIB = 2**20

#######################
#     Small files     #
#######################


def test_save_json_small_overwrite_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    benchmark(save_json, SMALL, path, exist_ok=True)
    assert load_json(path) == SMALL


def test_save_json_small_new_file_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    counter = itertools.count()
    benchmark(lambda: save_json(SMALL, tmp_path.joinpath(f"data-{next(counter)}.json")))
    assert load_json(tmp_path.joinpath("data-0.json")) == SMALL


def test_save_json_small_fsync_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    benchmark(save_json, SMALL, path, exist_ok=True, fsync=True)
    assert load_json(path) == SMALL


def test_save_text_small_overwrite_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    benchmark(save_text, "hello", path, exist_ok=True)
    assert load_text(path) == "hello"


def test_load_json_small_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json(SMALL, path)
    assert benchmark(load_json, path) == SMALL


def test_load_json_small_class_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json(SMALL, path)
    loader = JsonLoader()
    assert benchmark(loader.load, path) == SMALL


#######################
#     Large files     #
#######################


def test_save_json_large_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    benchmark(save_json, LARGE_JSON, path, exist_ok=True)
    assert load_json(path)[:3] == [0, 1, 2]


def test_load_json_large_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.json")
    save_json(LARGE_JSON, path)
    assert benchmark(load_json, path)[:3] == [0, 1, 2]


def test_save_pickle_1mib_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    benchmark(save_pickle, b"x" * ONE_MIB, path, exist_ok=True)
    assert len(load_pickle(path)) == ONE_MIB


def test_load_pickle_1mib_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.pkl")
    save_pickle(b"x" * ONE_MIB, path)
    assert len(benchmark(load_pickle, path)) == ONE_MIB


def test_save_text_1mib_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    benchmark(save_text, "x" * ONE_MIB, path, exist_ok=True)
    assert len(load_text(path)) == ONE_MIB


def test_load_text_1mib_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path.joinpath("data.txt")
    save_text("x" * ONE_MIB, path)
    assert len(benchmark(load_text, path)) == ONE_MIB


#########################
#     Many files        #
#########################


def save_many(directory: Path, num_files: int, run: int) -> None:
    for i in range(num_files):
        save_json(SMALL, directory.joinpath(f"run{run}-{i}.json"))


def test_save_json_100_files_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    counter = itertools.count()
    benchmark(lambda: save_many(tmp_path, 100, next(counter)))
    assert load_json(tmp_path.joinpath("run0-99.json")) == SMALL


##########################
#     Concurrent saves   #
##########################


def overwrite_same_path(path: Path, num_threads: int, num_saves: int) -> None:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                lambda: [save_json(SMALL, path, exist_ok=True) for _ in range(num_saves)]
            )
            for _ in range(num_threads)
        ]
        for future in futures:
            future.result()


def test_save_json_same_path_4_threads_benchmark(
    benchmark: BenchmarkFixture, tmp_path: Path
) -> None:
    path = tmp_path.joinpath("data.json")
    benchmark(overwrite_same_path, path, 4, 25)
    assert load_json(path) == SMALL


def distinct_paths_threads(directory: Path, num_threads: int, num_saves: int, run: int) -> None:
    def work(thread: int) -> None:
        for i in range(num_saves):
            save_json(SMALL, directory.joinpath(f"run{run}-{thread}-{i}.json"))

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for future in [executor.submit(work, t) for t in range(num_threads)]:
            future.result()


def test_save_json_distinct_paths_4_threads_benchmark(
    benchmark: BenchmarkFixture, tmp_path: Path
) -> None:
    counter = itertools.count()
    benchmark(lambda: distinct_paths_threads(tmp_path, 4, 25, next(counter)))
    assert load_json(tmp_path.joinpath("run0-3-24.json")) == SMALL


#####################
#     Resolvers     #
#####################


def test_resolve_loader_config_benchmark(benchmark: BenchmarkFixture) -> None:
    assert isinstance(benchmark(resolve_loader, {"_target_": "coola.io.JsonLoader"}), JsonLoader)


def test_resolve_saver_config_benchmark(benchmark: BenchmarkFixture) -> None:
    assert isinstance(benchmark(resolve_saver, {"_target_": "coola.io.JsonSaver"}), JsonSaver)


#####################
#     PyTorch       #
#####################


@torch_available
def test_save_torch_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    data = {"x": torch.arange(10_000)}
    benchmark(save_torch, data, tmp_path / "data.pt", exist_ok=True)


@torch_available
def test_load_torch_benchmark(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    path = tmp_path / "data.pt"
    save_torch({"x": torch.arange(10_000)}, path)
    assert benchmark(load_torch, path)["x"].shape == (10_000,)

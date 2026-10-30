r"""Benchmarks for the import time of ``coola`` and its subpackages.

Each import is measured in a fresh interpreter (``sys.modules`` caches
imports, so an in-process measurement only sees the first import). Run
them explicitly with:

    inv test.benchmark

The wall time of the subprocess includes the start of the interpreter,
so ``test_python_startup_benchmark`` gives the baseline to subtract. The
``extra_info`` of each result, saved with the benchmark JSON, also
records the import time measured inside the interpreter, the number of
modules loaded, the optional libraries loaded and the slowest imports
(from ``-X importtime``), to help finding what to optimize.
"""

from __future__ import annotations

import json
import pkgutil
import statistics
from typing import TYPE_CHECKING

import pytest

import coola
from tests.integration.helpers import run_interpreter

if TYPE_CHECKING:
    import subprocess

    from pytest_benchmark.fixture import BenchmarkFixture

OPTIONAL_LIBRARIES = ("jax", "numpy", "pandas", "polars", "pyarrow", "torch", "xarray")

SUBPACKAGES = sorted(m.name for m in pkgutil.iter_modules(coola.__path__) if m.ispkg)

TOP_N = 10

CODE = """
import json, sys, time
libraries = {libraries!r}
start = time.perf_counter()
{statement}
elapsed = time.perf_counter() - start
print(json.dumps({{
    "seconds": elapsed,
    "num_modules": len(sys.modules),
    "optional_libraries": sorted(x for x in libraries if x in sys.modules),
}}))
"""


def run_import(statement: str, *flags: str) -> subprocess.CompletedProcess[str]:
    r"""Run ``statement`` in a fresh isolated interpreter."""
    code = CODE.format(statement=statement, libraries=OPTIONAL_LIBRARIES)
    return run_interpreter("-I", *flags, "-c", code)


def slowest_imports(statement: str) -> list[dict[str, int | str]]:
    r"""Return the ``TOP_N`` imports with the largest cumulative
    time."""
    lines = run_import(statement, "-X", "importtime").stderr.splitlines()
    rows = []
    for line in lines:
        fields = line.removeprefix("import time:").split("|")
        if not line.startswith("import time:") or len(fields) != 3 or "cumulative" in line:
            continue
        rows.append({"module": fields[2].strip(), "cumulative_us": int(fields[1])})
    return sorted(rows, key=lambda row: row["cumulative_us"], reverse=True)[:TOP_N]


def benchmark_import(
    benchmark: BenchmarkFixture, statement: str, import_results: dict[str, dict]
) -> None:
    r"""Benchmark ``statement`` and save the details in
    ``extra_info``."""
    results = []

    def target() -> None:
        results.append(json.loads(run_import(statement).stdout))

    benchmark.pedantic(target, rounds=10, warmup_rounds=1, iterations=1)
    seconds = [r["seconds"] for r in results[1:]]  # the first run is the warm-up
    import_results[statement] = {
        "min": min(seconds),
        "median": statistics.median(seconds),
        "num_modules": results[-1]["num_modules"],
        "optional_libraries": results[-1]["optional_libraries"],
    }
    benchmark.extra_info.update(
        {
            "import_seconds_min": min(seconds),
            "import_seconds_median": statistics.median(seconds),
            "num_modules": results[-1]["num_modules"],
            "optional_libraries": results[-1]["optional_libraries"],
            "slowest_imports": slowest_imports(statement),
        }
    )


def test_python_startup_benchmark(
    benchmark: BenchmarkFixture, import_results: dict[str, dict]
) -> None:
    benchmark_import(benchmark, "pass", import_results)


def test_import_coola_benchmark(
    benchmark: BenchmarkFixture, import_results: dict[str, dict]
) -> None:
    benchmark_import(benchmark, "import coola", import_results)


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_from_coola_import_subpackage_benchmark(
    benchmark: BenchmarkFixture, import_results: dict[str, dict], subpackage: str
) -> None:
    benchmark_import(benchmark, f"from coola import {subpackage}", import_results)

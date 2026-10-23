r"""Define the tasks that are executed with invoke."""

from __future__ import annotations

from invoke import Context, Task
from invoke_tasklib import ns

__all__ = ["ns"]


def benchmark_import(c: Context) -> None:
    r"""Run the import time benchmarks."""
    c.run(
        "python -m pytest tests/benchmarks/test_import_benchmark.py --benchmark-only",
        pty=True,
    )


ns.collections["test"].add_task(Task(benchmark_import, name="benchmark-import"))

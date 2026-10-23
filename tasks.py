r"""Define the tasks that are executed with invoke."""

from __future__ import annotations

from invoke import Collection, Context, task
from invoke_tasklib import ns

__all__ = ["ns"]


@task
def import_time(c: Context) -> None:
    r"""Run the import time benchmarks."""
    c.run(
        "python -m pytest tests/benchmarks/test_import_benchmark.py --benchmark-only",
        pty=True,
    )


ns.add_collection(Collection("benchmark", import_time))

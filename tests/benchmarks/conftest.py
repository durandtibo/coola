r"""Skip the benchmarks unless they are explicitly requested.

The benchmarks take several seconds, so they are not run with the other
tests. Run them with ``inv test.benchmark`` (``--benchmark-only``).
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("benchmark_only"):
        return
    skip = pytest.mark.skip(reason="benchmarks only run with --benchmark-only")
    for item in items:
        if "tests/benchmarks" in item.nodeid:
            item.add_marker(skip)

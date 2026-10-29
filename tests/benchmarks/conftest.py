r"""Skip the benchmarks unless they are explicitly requested.

The benchmarks take several seconds, so they are not run with the other
tests. Run them with ``inv test.benchmark`` (``--benchmark-only``).
"""

from __future__ import annotations

import importlib

import pytest

IMPORT_RESULTS = pytest.StashKey[dict[str, dict]]()


@pytest.fixture
def import_results(request: pytest.FixtureRequest) -> dict[str, dict]:
    r"""Return the dict where the import benchmarks save their
    results."""
    return request.config.stash.setdefault(IMPORT_RESULTS, {})


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    r"""Show the import time of each import in a table."""
    results = terminalreporter.config.stash.get(IMPORT_RESULTS, {})
    if not results:
        return
    try:
        console_cls = importlib.import_module("rich.console").Console
        table_cls = importlib.import_module("rich.table").Table
    except ImportError:
        return
    table = table_cls(title="Import time (ms, measured inside a fresh interpreter)")
    table.add_column("Import", no_wrap=True)
    for name in ("Min", "Median"):
        table.add_column(name, justify="right")
    table.add_column("Modules", justify="right")
    table.add_column("Optional libraries")
    for statement, r in sorted(results.items(), key=lambda item: item[1]["median"]):
        table.add_row(
            statement,
            f"{r['min'] * 1000:.1f}",
            f"{r['median'] * 1000:.1f}",
            str(r["num_modules"]),
            ", ".join(r["optional_libraries"]),
        )
    console_cls().print(table)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("benchmark_only"):
        return
    skip = pytest.mark.skip(reason="benchmarks only run with --benchmark-only")
    for item in items:
        if "tests/benchmarks" in item.nodeid:
            item.add_marker(skip)

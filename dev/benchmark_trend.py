# noqa: INP001
r"""Script to show the trend of the benchmarks over the past days.

It reads the pytest-benchmark results saved by the nightly benchmark
workflow, deletes the results older than the window, and prints a Markdown
table.

Example:
    python dev/benchmark_trend.py .benchmarks --days 90
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)

SPARKS = "▁▂▃▄▅▆▇█"


def get_label(benchmark: dict) -> str:
    r"""Get the label of a benchmark.

    Args:
        benchmark: A benchmark entry of a pytest-benchmark JSON file.

    Returns:
        The name of the benchmark without the ``test_`` prefix and the
            ``_benchmark`` suffix.
    """
    return benchmark["name"].removeprefix("test_").removesuffix("_benchmark")


def load_runs(storage: Path, days: int, *, prune: bool = False) -> list[tuple[datetime, dict]]:
    r"""Load the results of the runs in the window.

    Args:
        storage: The pytest-benchmark storage directory.
        days: The size of the window in days.
        prune: If ``True``, delete the files older than the window.

    Returns:
        The runs sorted by date, each one as ``(date, {label: median
            time in seconds})``.
    """
    limit = datetime.now(tz=UTC) - timedelta(days=days)
    runs = []
    for path in sorted(storage.rglob("*.json")):
        data = json.loads(path.read_text())
        date = datetime.fromisoformat(data["datetime"])
        if date.tzinfo is None:
            date = date.replace(tzinfo=UTC)
        if date < limit:
            if prune:
                path.unlink()
            continue
        times = {get_label(b): b["stats"]["median"] for b in data["benchmarks"]}
        runs.append((date, times))
    return sorted(runs, key=lambda run: run[0])


def sparkline(values: list[float]) -> str:
    r"""Draw the values with block characters.

    Args:
        values: The values to draw.

    Returns:
        One character per value, scaled between the min and the max.
    """
    low, high = min(values), max(values)
    if high == low:
        return SPARKS[0] * len(values)
    return "".join(SPARKS[round((v - low) / (high - low) * (len(SPARKS) - 1))] for v in values)


def make_table(runs: list[tuple[datetime, dict]]) -> str:
    r"""Make the Markdown table of the trend of each benchmark.

    Args:
        runs: The runs returned by ``load_runs``.

    Returns:
        A Markdown table sorted by latest time.
    """
    series: dict[str, list[float]] = {}
    for _, times in runs:
        for label, seconds in times.items():
            series.setdefault(label, []).append(seconds * 1e6)
    lines = [
        (
            "| Benchmark | Latest (µs) | First (µs) | Change | Median (µs) | Min (µs) | Max (µs) "
            "| Runs | Trend |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, values in sorted(series.items(), key=lambda item: item[1][-1]):
        change = (values[-1] - values[0]) / values[0] * 100 if values[0] else 0.0
        lines.append(
            f"| {label} | {values[-1]:.1f} | {values[0]:.1f} | {change:+.1f}% "
            f"| {statistics.median(values):.1f} | {min(values):.1f} | {max(values):.1f} "
            f"| {len(values)} | `{sparkline(values)}` |"
        )
    return "\n".join(lines)


def main() -> None:
    r"""Print the trend table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storage", type=Path, help="pytest-benchmark storage directory")
    parser.add_argument("--days", type=int, default=90, help="size of the window in days")
    parser.add_argument("--prune", action="store_true", help="delete the older results")
    args = parser.parse_args()
    runs = load_runs(args.storage, args.days, prune=args.prune)
    if not runs:
        logger.info("No results.")
        return
    sys.stdout.write(
        f"## Benchmarks over the last {args.days} days ({len(runs)} runs)\n\n{make_table(runs)}\n"
    )


if __name__ == "__main__":
    # The table is written to stdout so it can be piped into a file.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()

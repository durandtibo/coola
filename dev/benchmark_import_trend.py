# noqa: INP001
r"""Script to show the trend of the import time over the past days.

It reads the pytest-benchmark results saved by the daily import time
benchmark workflow, deletes the results older than the window, and prints a
Markdown table.

Example:
    python dev/benchmark_import_trend.py .benchmarks --days 90
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
UNIT = "ms"  # unit of the times shown in the reports
SCALE = 1000  # conversion from seconds to UNIT


def get_label(benchmark: dict) -> str:
    r"""Get the label of a benchmark.

    Args:
        benchmark: A benchmark entry of a pytest-benchmark JSON file.

    Returns:
        The subpackage name for ``from coola import <subpackage>``, or the
            name of the benchmark without the ``test_`` prefix and the
            ``_benchmark`` suffix.
    """
    if benchmark.get("param"):
        return benchmark["param"]
    return benchmark["name"].removeprefix("test_").removesuffix("_benchmark")


def load_runs(storage: Path, days: int, *, prune: bool = False) -> list[tuple[datetime, dict]]:
    r"""Load the results of the runs in the window.

    Args:
        storage: The pytest-benchmark storage directory.
        days: The size of the window in days.
        prune: If ``True``, delete the files older than the window.

    Returns:
        The runs sorted by date, each one as ``(date, {label: median
            import time in seconds})``.
    """
    limit = datetime.now(tz=UTC) - timedelta(days=days)
    runs = []
    for path in sorted(storage.rglob("*.json")):
        data = json.loads(path.read_text())
        date = datetime.fromisoformat(data["datetime"])
        if date < limit:
            if prune:
                path.unlink()
            continue
        times = {
            get_label(b): b["extra_info"]["import_seconds_median"]
            for b in data["benchmarks"]
            if "import_seconds_median" in b.get("extra_info", {})
        }
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
    r"""Make the Markdown table of the trend of each import.

    Args:
        runs: The runs returned by ``load_runs``.

    Returns:
        A Markdown table sorted by latest import time.
    """
    series: dict[str, list[float]] = {}
    for _, times in runs:
        for label, seconds in times.items():
            series.setdefault(label, []).append(seconds * 1000)
    lines = [
        (
            "| Import | Latest (ms) | First (ms) | Change | Median (ms) | Min (ms) | Max (ms) "
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


def find_regressions(
    runs: list[tuple[datetime, dict]], window: int = 7, threshold: float = 50.0
) -> list[tuple[str, float, float, float]]:
    r"""Find the imports slower than their recent median.

    The latest run is compared with the median of the previous runs
    (at most ``window``), which is more stable than the comparison with
    one run because the runners are noisy.

    Args:
        runs: The runs returned by ``load_runs``.
        window: The maximum number of previous runs used for the median.
        threshold: The slowdown in percent above which a benchmark is
            a regression.

    Returns:
        The regressions sorted by decreasing change, each one as
            ``(label, latest, median of the previous runs, change in
            percent)``. It is empty if there are fewer than two runs.
    """
    if len(runs) < 2:
        return []
    regressions = []
    for label, latest in runs[-1][1].items():
        previous = [times[label] for _, times in runs[:-1] if label in times][-window:]
        if not previous:
            continue
        baseline = statistics.median(previous)
        change = (latest - baseline) / baseline * 100 if baseline else 0.0
        if change > threshold:
            regressions.append((label, latest, baseline, change))
    return sorted(regressions, key=lambda item: item[3], reverse=True)


def make_regression_report(regressions: list[tuple[str, float, float, float]]) -> str:
    r"""Make the Markdown report of the regressions.

    Args:
        regressions: The regressions returned by ``find_regressions``.

    Returns:
        A Markdown section, or an empty string if there is no regression.
    """
    if not regressions:
        return ""
    lines = [
        "### Regressions",
        "",
        f"| Benchmark | Latest ({UNIT}) | Previous median ({UNIT}) | Change |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {label} | {latest * SCALE:.1f} | {baseline * SCALE:.1f} | {change:+.1f}% |"
        for label, latest, baseline, change in regressions
    )
    return "\n".join(lines)


def make_annotations(regressions: list[tuple[str, float, float, float]]) -> list[str]:
    r"""Make the GitHub Actions warning annotations of the regressions.

    Args:
        regressions: The regressions returned by ``find_regressions``.

    Returns:
        One ``::warning::`` workflow command per regression.
    """
    return [
        f"::warning title=Regression::{label} is {change:.1f}% slower than the median of the "
        f"previous runs ({latest * SCALE:.1f} {UNIT} instead of {baseline * SCALE:.1f} {UNIT})"
        for label, latest, baseline, change in regressions
    ]


def main() -> None:
    r"""Print the trend table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("storage", type=Path, help="pytest-benchmark storage directory")
    parser.add_argument("--days", type=int, default=90, help="size of the window in days")
    parser.add_argument("--prune", action="store_true", help="delete the older results")
    parser.add_argument(
        "--window", type=int, default=7, help="number of previous runs used to detect regressions"
    )
    parser.add_argument(
        "--threshold", type=float, default=50.0, help="slowdown in percent of a regression"
    )
    args = parser.parse_args()
    runs = load_runs(args.storage, args.days, prune=args.prune)
    if not runs:
        logger.info("No results.")
        return
    regressions = find_regressions(runs, window=args.window, threshold=args.threshold)
    # GitHub Actions turns the lines starting with ::warning:: into annotations of the run.
    for annotation in make_annotations(regressions):
        sys.stderr.write(f"{annotation}\n")
    report = make_regression_report(regressions)
    sys.stdout.write(
        f"## Import time over the last {args.days} days ({len(runs)} runs)\n\n{make_table(runs)}\n"
    )
    if report:
        sys.stdout.write(f"\n{report}\n")


if __name__ == "__main__":
    # The table is written to stdout so it can be piped into a file.
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()

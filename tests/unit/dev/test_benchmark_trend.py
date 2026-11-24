from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "dev" / "benchmark_trend.py"
SPEC = importlib.util.spec_from_file_location("benchmark_trend", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

DATE = datetime(2026, 1, 1, tzinfo=UTC)


def make_runs(*values: dict) -> list[tuple[datetime, dict]]:
    return [(DATE, times) for times in values]


######################################
#     Tests for find_regressions     #
######################################


def test_find_regressions_slower_than_threshold() -> None:
    runs = make_runs({"a": 1.0}, {"a": 1.0}, {"a": 1.0}, {"a": 2.0})
    assert module.find_regressions(runs) == [("a", 2.0, 1.0, 100.0)]


def test_find_regressions_at_threshold_is_not_a_regression() -> None:
    runs = make_runs({"a": 1.0}, {"a": 1.5})
    assert module.find_regressions(runs) == []


def test_find_regressions_faster_is_not_a_regression() -> None:
    runs = make_runs({"a": 2.0}, {"a": 1.0})
    assert module.find_regressions(runs) == []


def test_find_regressions_custom_threshold() -> None:
    runs = make_runs({"a": 1.0}, {"a": 1.5})
    assert module.find_regressions(runs, threshold=10.0) == [("a", 1.5, 1.0, 50.0)]


def test_find_regressions_uses_median_of_previous_runs() -> None:
    # The median of [1, 1, 10] is 1, so one slow run in the history does not hide a regression.
    runs = make_runs({"a": 1.0}, {"a": 1.0}, {"a": 10.0}, {"a": 2.0})
    assert module.find_regressions(runs) == [("a", 2.0, 1.0, 100.0)]


def test_find_regressions_uses_only_window_runs() -> None:
    runs = make_runs({"a": 100.0}, {"a": 1.0}, {"a": 1.0}, {"a": 2.0})
    assert module.find_regressions(runs, window=2) == [("a", 2.0, 1.0, 100.0)]


def test_find_regressions_one_run() -> None:
    assert module.find_regressions(make_runs({"a": 1.0})) == []


def test_find_regressions_no_run() -> None:
    assert module.find_regressions([]) == []


def test_find_regressions_new_benchmark_is_ignored() -> None:
    runs = make_runs({"a": 1.0}, {"a": 1.0, "b": 5.0})
    assert module.find_regressions(runs) == []


def test_find_regressions_benchmark_missing_in_latest_run_is_ignored() -> None:
    runs = make_runs({"a": 1.0, "b": 1.0}, {"a": 1.0})
    assert module.find_regressions(runs) == []


def test_find_regressions_zero_baseline() -> None:
    runs = make_runs({"a": 0.0}, {"a": 1.0})
    assert module.find_regressions(runs) == []


def test_find_regressions_sorted_by_decreasing_change() -> None:
    runs = make_runs({"a": 1.0, "b": 1.0}, {"a": 2.0, "b": 3.0})
    assert module.find_regressions(runs) == [("b", 3.0, 1.0, 200.0), ("a", 2.0, 1.0, 100.0)]


def test_find_regressions_does_not_modify_runs() -> None:
    runs = make_runs({"a": 1.0}, {"a": 2.0})
    module.find_regressions(runs)
    assert runs == make_runs({"a": 1.0}, {"a": 2.0})


############################################
#     Tests for make_regression_report     #
############################################


def test_make_regression_report_empty() -> None:
    assert module.make_regression_report([]) == ""


def test_make_regression_report() -> None:
    assert module.make_regression_report([("a", 2e-06, 1e-06, 100.0)]) == (
        "### Regressions\n"
        "\n"
        "| Benchmark | Latest (µs) | Previous median (µs) | Change |\n"
        "|---|---:|---:|---:|\n"
        "| a | 2.0 | 1.0 | +100.0% |"
    )


######################################
#     Tests for make_annotations     #
######################################


def test_make_annotations_empty() -> None:
    assert module.make_annotations([]) == []


def test_make_annotations() -> None:
    assert module.make_annotations([("a", 2e-06, 1e-06, 100.0)]) == [
        (
            "::warning title=Regression::a is 100.0% slower than the median of the previous runs "
            "(2.0 µs instead of 1.0 µs)"
        )
    ]

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "dev" / "benchmark_trend.py"
SPEC = importlib.util.spec_from_file_location("benchmark_trend", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

DATE = datetime(2026, 1, 1, tzinfo=UTC)
BENCH_A = {"name": "test_a_benchmark", "stats": {"median": 0.001}}
BENCH_B = {"name": "test_b_benchmark", "stats": {"median": 0.002}}


def make_runs(*values: dict) -> list[tuple[datetime, dict]]:
    return [(DATE, times) for times in values]


def write_run(path: Path, date: datetime, benchmarks: list[dict]) -> None:
    path.write_text(json.dumps({"datetime": date.isoformat(), "benchmarks": benchmarks}))


def write_regression(path: Path) -> None:
    now = datetime.now(tz=UTC)
    slow = {**BENCH_A, "stats": {"median": 0.002}, "extra_info": {"import_seconds_median": 0.002}}
    write_run(path / "1.json", now - timedelta(days=2), [BENCH_A])
    write_run(path / "2.json", now - timedelta(days=1), [slow])


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


###############################
#     Tests for get_label     #
###############################


def test_get_label_strips_prefix_and_suffix() -> None:
    assert module.get_label({"name": "test_foo_benchmark"}) == "foo"


def test_get_label_without_prefix_and_suffix() -> None:
    assert module.get_label({"name": "foo"}) == "foo"


def test_get_label_strips_only_one_prefix_and_suffix() -> None:
    assert module.get_label({"name": "test_test_benchmark_benchmark"}) == "test_benchmark"


###############################
#     Tests for sparkline     #
###############################


def test_sparkline_increasing() -> None:
    assert module.sparkline([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) == "▁▂▃▄▅▆▇█"


def test_sparkline_constant() -> None:
    assert module.sparkline([2.0, 2.0, 2.0]) == "▁▁▁"


def test_sparkline_single_value() -> None:
    assert module.sparkline([5.0]) == "▁"


def test_sparkline_min_and_max() -> None:
    assert module.sparkline([3.0, 1.0, 2.0]) == "█▁▅"


###############################
#     Tests for load_runs     #
###############################


def test_load_runs_empty_storage(tmp_path: Path) -> None:
    assert module.load_runs(tmp_path, days=90) == []


def test_load_runs_sorted_by_date(tmp_path: Path) -> None:
    now = datetime.now(tz=UTC)
    write_run(tmp_path / "a.json", now - timedelta(days=1), [BENCH_B])
    write_run(tmp_path / "b.json", now - timedelta(days=2), [BENCH_A])
    assert module.load_runs(tmp_path, days=90) == [
        (now - timedelta(days=2), {"a": 0.001}),
        (now - timedelta(days=1), {"b": 0.002}),
    ]


def test_load_runs_skips_old_runs(tmp_path: Path) -> None:
    now = datetime.now(tz=UTC)
    write_run(tmp_path / "old.json", now - timedelta(days=10), [BENCH_A])
    write_run(tmp_path / "new.json", now - timedelta(days=1), [BENCH_B])
    assert module.load_runs(tmp_path, days=5) == [(now - timedelta(days=1), {"b": 0.002})]


def test_load_runs_keeps_old_files_without_prune(tmp_path: Path) -> None:
    write_run(tmp_path / "old.json", datetime.now(tz=UTC) - timedelta(days=10), [BENCH_A])
    module.load_runs(tmp_path, days=5)
    assert (tmp_path / "old.json").exists()


def test_load_runs_prune_deletes_old_files(tmp_path: Path) -> None:
    write_run(tmp_path / "old.json", datetime.now(tz=UTC) - timedelta(days=10), [BENCH_A])
    module.load_runs(tmp_path, days=5, prune=True)
    assert not (tmp_path / "old.json").exists()


def test_load_runs_prune_keeps_recent_files(tmp_path: Path) -> None:
    write_run(tmp_path / "new.json", datetime.now(tz=UTC) - timedelta(days=1), [BENCH_A])
    module.load_runs(tmp_path, days=5, prune=True)
    assert (tmp_path / "new.json").exists()


def test_load_runs_searches_subdirectories(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    date = datetime.now(tz=UTC) - timedelta(days=1)
    write_run(sub / "a.json", date, [BENCH_A])
    assert module.load_runs(tmp_path, days=5) == [(date, {"a": 0.001})]


def test_load_runs_ignores_non_json_files(tmp_path: Path) -> None:
    tmp_path.joinpath("notes.txt").write_text("not json")
    assert module.load_runs(tmp_path, days=5) == []


def test_load_runs_naive_datetime_is_utc(tmp_path: Path) -> None:
    date = datetime.now(tz=UTC).replace(tzinfo=None) - timedelta(days=1)
    write_run(tmp_path / "a.json", date, [BENCH_A])
    assert module.load_runs(tmp_path, days=5) == [(date.replace(tzinfo=UTC), {"a": 0.001})]


################################
#     Tests for make_table     #
################################


def test_make_table_empty() -> None:
    assert module.make_table([]) == (
        "| Benchmark | Latest (µs) | First (µs) | Change | Median (µs) | Min (µs) "
        "| Max (µs) | Runs | Trend |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )


def test_make_table_one_benchmark() -> None:
    runs = make_runs({"a": 1.0}, {"a": 2.0}, {"a": 3.0})
    assert module.make_table(runs) == (
        "| Benchmark | Latest (µs) | First (µs) | Change | Median (µs) | Min (µs) "
        "| Max (µs) | Runs | Trend |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|\n"
        "| a | 3000000.0 | 1000000.0 | +200.0% | 2000000.0 | 1000000.0 | 3000000.0 | 3 | `▁▅█` |"
    )


def test_make_table_sorted_by_latest_time() -> None:
    runs = make_runs({"a": 2.0, "b": 1.0})
    assert module.make_table(runs).splitlines()[2:] == [
        "| b | 1000000.0 | 1000000.0 | +0.0% | 1000000.0 | 1000000.0 | 1000000.0 | 1 | `▁` |",
        "| a | 2000000.0 | 2000000.0 | +0.0% | 2000000.0 | 2000000.0 | 2000000.0 | 1 | `▁` |",
    ]


def test_make_table_benchmark_missing_in_some_runs() -> None:
    runs = make_runs({"a": 1.0}, {"a": 1.0, "b": 2.0})
    assert module.make_table(runs).splitlines()[-1] == (
        "| b | 2000000.0 | 2000000.0 | +0.0% | 2000000.0 | 2000000.0 | 2000000.0 | 1 | `▁` |"
    )


def test_make_table_zero_first_value_has_zero_change() -> None:
    runs = make_runs({"a": 0.0}, {"a": 1.0})
    assert module.make_table(runs).splitlines()[-1] == (
        "| a | 1000000.0 | 0.0 | +0.0% | 500000.0 | 0.0 | 1000000.0 | 2 | `▁█` |"
    )


##########################
#     Tests for main     #
##########################


def run_main(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["script", *args])
    module.main()


def test_main_no_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        run_main(monkeypatch, str(tmp_path))
    assert "No results." in caplog.messages


def test_main_no_results_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run_main(monkeypatch, str(tmp_path))
    assert capsys.readouterr().out == ""


def test_main_prints_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    write_run(tmp_path / "a.json", datetime.now(tz=UTC) - timedelta(days=1), [BENCH_A])
    run_main(monkeypatch, str(tmp_path), "--days", "5")
    assert capsys.readouterr().out == (
        f"## Benchmarks over the last 5 days (1 runs)\n\n{module.make_table(module.load_runs(tmp_path, 5))}\n"
    )


def test_main_without_regression_has_no_annotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    write_run(tmp_path / "a.json", datetime.now(tz=UTC) - timedelta(days=1), [BENCH_A])
    run_main(monkeypatch, str(tmp_path))
    assert capsys.readouterr().err == ""


def test_main_regression_prints_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    write_regression(tmp_path)
    run_main(monkeypatch, str(tmp_path))
    assert "### Regressions" in capsys.readouterr().out


def test_main_regression_prints_annotation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    write_regression(tmp_path)
    run_main(monkeypatch, str(tmp_path))
    assert capsys.readouterr().err.startswith("::warning title=Regression::a is 100.0% slower")


def test_main_threshold_hides_regression(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    write_regression(tmp_path)
    run_main(monkeypatch, str(tmp_path), "--threshold", "150")
    assert "### Regressions" not in capsys.readouterr().out


def test_main_prune_deletes_old_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_run(tmp_path / "old.json", datetime.now(tz=UTC) - timedelta(days=10), [BENCH_A])
    run_main(monkeypatch, str(tmp_path), "--days", "5", "--prune")
    assert not (tmp_path / "old.json").exists()

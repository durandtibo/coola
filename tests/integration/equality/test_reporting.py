r"""Check that the explanation of the difference is recorded once, and
that nothing is logged."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality import compare, objects_are_equal
from coola.equality.config import EqualityConfig
from coola.equality.interface import run_comparison
from coola.equality.tester import get_default_registry

if TYPE_CHECKING:
    import pytest

ACTUAL = {"a": [1, 2, 3]}
EXPECTED = {"a": [1, 2, 4]}


def _reports() -> list[tuple[str, tuple[object, ...]]]:
    config = EqualityConfig(reports=[])
    run_comparison(get_default_registry(), ACTUAL, EXPECTED, config)
    return config.reports or []


def test_reports_the_innermost_path_first() -> None:
    assert [path for _, path in _reports()] == [(), (2,), ("a",), ()]


def test_compare_reports_the_reason_of_the_innermost_difference() -> None:
    assert compare(ACTUAL, EXPECTED).reason == _reports()[0][0]


def test_compare_path_assembles_the_reports() -> None:
    assert compare(ACTUAL, EXPECTED).path == ("a", 2)


def test_objects_are_equal_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        objects_are_equal(ACTUAL, EXPECTED)
    assert not caplog.records


def test_compare_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        compare(ACTUAL, EXPECTED)
    assert not caplog.records


def test_objects_are_equal_records_no_report() -> None:
    config = EqualityConfig()
    run_comparison(get_default_registry(), ACTUAL, EXPECTED, config)
    assert config.reports is None

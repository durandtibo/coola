r"""Check that the logging and the reporter receive the same
difference."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality import compare, objects_are_equal

if TYPE_CHECKING:
    import pytest

ACTUAL = {"a": [1, 2, 3]}
EXPECTED = {"a": [1, 2, 4]}


def test_show_difference_logs_on_the_equality_logger(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coola.equality"):
        objects_are_equal(ACTUAL, EXPECTED, show_difference=True)
    assert {record.name for record in caplog.records} == {"coola.equality"}


def test_show_difference_logs_the_innermost_path_first(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coola.equality"):
        objects_are_equal(ACTUAL, EXPECTED, show_difference=True)
    assert [record.coola_path for record in caplog.records] == [(), (2,), ("a",)]


def test_show_difference_and_compare_report_the_same_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="coola.equality"):
        objects_are_equal(ACTUAL, EXPECTED, show_difference=True)
    assert caplog.records[0].getMessage() == compare(ACTUAL, EXPECTED).reason

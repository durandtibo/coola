r"""Check that the explanation of the difference is recorded once, and
that nothing is logged."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coola.equality import compare, objects_are_equal

if TYPE_CHECKING:
    import pytest

ACTUAL = {"a": [1, 2, 3]}
EXPECTED = {"a": [1, 2, 4]}


def test_compare_reports_one_difference_with_the_whole_path() -> None:
    assert compare(ACTUAL, EXPECTED).path == ("a", 2)


def test_compare_reports_the_reason_of_the_innermost_difference() -> None:
    assert (
        compare(ACTUAL, EXPECTED).reason == "numbers are different:\n  actual   : 3\n  expected : 4"
    )


def test_objects_are_equal_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        objects_are_equal(ACTUAL, EXPECTED)
    assert not caplog.records


def test_compare_does_not_log(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        compare(ACTUAL, EXPECTED)
    assert not caplog.records

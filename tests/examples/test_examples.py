from __future__ import annotations

import logging
from pathlib import Path

import pytest
from examples import hashing, identifier, io, jsonable, nested, rounding, summary

from tests.integration.helpers import run_python

EXAMPLES = sorted(
    path
    for path in Path(__file__).parents[2].joinpath("examples").glob("*.py")
    if path.name != "__init__.py"
)

##################################
#     Tests for the examples     #
##################################


def test_examples_are_found() -> None:
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_runs_without_error(path: Path) -> None:
    # The examples are scripts: run them in a fresh interpreter so they cannot rot silently.
    run_python(f"import runpy; runpy.run_path({str(path)!r}, run_name='__main__')")


def test_hashing_example_logs_that_the_hashes_behave_as_expected(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        hashing.main()
    assert caplog.messages.count("same hash for the same content: True") == 1


def test_hashing_example_logs_that_the_content_id_has_the_right_length(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        hashing.main()
    assert "content id has 8 characters: True" in caplog.messages


def test_nested_example_logs_the_flat_keys(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        nested.main()
    assert caplog.messages[0].startswith("flat keys: ['lr', 'model.layers.0.units'")


def test_nested_example_logs_that_the_round_trip_works(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        nested.main()
    assert "round trip: True" in caplog.messages


def test_summary_example_logs_the_median(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        summary.main()
    assert "median: 2" in caplog.messages


def test_jsonable_example_logs_the_sorted_set(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        jsonable.main()
    assert "jsonable: {'tags': ['a', 'b'], 'point': [1, 2], 'name': 'coola'}" in caplog.messages


def test_rounding_example_logs_the_rounded_values(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        rounding.main()
    assert "rounded: {'a': 2.6, 'b': [1.2, 5.7]}" in caplog.messages


def test_identifier_example_logs_that_the_content_id_is_stable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        identifier.main()
    assert "same content id for the same content: True" in caplog.messages


def test_io_example_logs_that_the_round_trip_works(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        io.main()
    assert "round trip: True" in caplog.messages

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.helpers import run_python

EXAMPLES = sorted(
    path
    for path in Path(__file__).parents[2].joinpath("examples").glob("*.py")
    if path.name != "__init__.py"
)

####################################
#     Tests for the examples       #
####################################


def test_examples_are_found() -> None:
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_runs_without_error(path: Path) -> None:
    # The examples are scripts: run them in a fresh interpreter so they cannot rot silently.
    run_python(f"import runpy; runpy.run_path({str(path)!r}, run_name='__main__')")

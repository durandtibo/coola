from __future__ import annotations

from pathlib import Path

import pytest

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


def run_example(name: str) -> list[str]:
    path = Path(__file__).parents[2].joinpath("examples", name)
    return run_python(f"import runpy; runpy.run_path({str(path)!r}, run_name='__main__')")


def test_hashing_example_prints_that_the_hashes_behave_as_expected() -> None:
    assert run_example("hashing.py").count("True") == 3


def test_nested_example_prints_the_flat_keys() -> None:
    assert "'model.layers.0.units'," in run_example("nested.py")


def test_nested_example_prints_that_the_round_trip_works() -> None:
    assert run_example("nested.py")[run_example("nested.py").index("trip:") + 1] == "True"


def test_summary_example_prints_the_median() -> None:
    assert run_example("summary.py")[-3:] == ["median:", "2", "max:", "3"][-3:]

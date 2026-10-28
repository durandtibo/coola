from __future__ import annotations

import pytest

from coola.testing import marks
from coola.utils import imports
from tests.integration.helpers import run_python

HEADER = "import pytest\nfrom coola.testing.marks import requires\n"

###########################################
#     Tests for the outcome of a mark     #
###########################################


def test_marks_on_functions_pass_and_skip(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        HEADER
        + """
yes_mark, _ = requires(True, "X")
_, not_available = requires(True, "X")


@yes_mark
def test_runs(): ...


@not_available
def test_skipped(): ...
"""
    )
    pytester.runpytest().assert_outcomes(passed=1, skipped=1)


def test_skip_reason_is_reported(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        HEADER
        + """
available, _ = requires(False, "Dask")


@available
def test_skipped(): ...
"""
    )
    assert "Requires Dask" in pytester.runpytest("-rs").stdout.str()


def test_marks_on_a_module(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        HEADER
        + """
available, _ = requires(False, "X")
pytestmark = available


def test_one(): ...


def test_two(): ...
"""
    )
    pytester.runpytest().assert_outcomes(skipped=2)


def test_marks_on_a_class(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        HEADER
        + """
_, not_available = requires(True, "X")


@not_available
class TestSkipped:
    def test_one(self): ...


def test_runs(): ...
"""
    )
    pytester.runpytest().assert_outcomes(passed=1, skipped=1)


def test_marks_in_pytest_param(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        HEADER
        + """
available, _ = requires(False, "X")


@pytest.mark.parametrize("value", [1, pytest.param(2, marks=available)])
def test_value(value): ...
"""
    )
    pytester.runpytest().assert_outcomes(passed=1, skipped=1)


def test_real_mark_is_importable_in_a_generated_module(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
from coola.testing.marks import numpy_available, numpy_not_available


@numpy_available
def test_with_numpy(): ...


@numpy_not_available
def test_without_numpy(): ...
"""
    )
    pytester.runpytest().assert_outcomes(passed=1, skipped=1)


####################################
#     Tests for the lazy marks     #
####################################


def test_importing_the_marks_imports_no_optional_library() -> None:
    code = (
        "import sys, coola.testing.marks; "
        "print(*('torch' in sys.modules, 'jax' in sys.modules, 'numpy' in sys.modules, "
        "'pandas' in sys.modules))"
    )
    assert run_python(code) == ["False", "False", "False", "False"]


def test_accessing_a_mark_does_not_import_the_other_libraries() -> None:
    code = (
        "import sys; from coola.testing.marks import numpy_available; "
        "print('torch' in sys.modules, 'jax' in sys.modules)"
    )
    assert run_python(code) == ["False", "False"]


##############################################
#     Tests for the agreement with coola     #
##############################################


@pytest.mark.parametrize(
    "key",
    [key for key in marks._DEPENDENCIES if key not in {"torch_cuda", "torch_mps"}],
)
def test_mark_agrees_with_the_library(key: str) -> None:
    is_available = getattr(imports, f"is_{key}_available")()
    assert getattr(marks, f"{key}_available").args[0] is (not is_available)

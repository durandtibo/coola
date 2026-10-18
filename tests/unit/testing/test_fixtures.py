from __future__ import annotations

import pytest

from coola.testing import fixtures


@pytest.mark.parametrize(
    "available_name",
    [
        "colorlog",
        "jax",
        "numpy",
        "packaging",
        "pandas",
        "polars",
        "pyarrow",
        "pydantic",
        "rich",
        "torch",
        "torch_cuda",
        "torch_mps",
        "torch_numpy",
        "xarray",
    ],
)
def test_available_fixture_has_not_available_counterpart(available_name: str) -> None:
    r"""Every ``*_available`` fixture must have a ``*_not_available``
    counterpart."""
    available = getattr(fixtures, f"{available_name}_available")
    not_available = getattr(fixtures, f"{available_name}_not_available")
    assert isinstance(available, pytest.MarkDecorator)
    assert isinstance(not_available, pytest.MarkDecorator)


@pytest.mark.parametrize(
    "name",
    [
        "torch_cuda_not_available",
        "torch_mps_not_available",
        "torch_numpy_not_available",
    ],
)
def test_all_exports_new_not_available_fixtures(name: str) -> None:
    assert name in fixtures.__all__


def test_torch_cuda_not_available_is_opposite_of_torch_cuda_available() -> None:
    cuda_available = fixtures.torch_cuda_available.args[0]
    cuda_not_available = fixtures.torch_cuda_not_available.args[0]
    assert cuda_available != cuda_not_available


def test_torch_mps_not_available_is_opposite_of_torch_mps_available() -> None:
    mps_available = fixtures.torch_mps_available.args[0]
    mps_not_available = fixtures.torch_mps_not_available.args[0]
    assert mps_available != mps_not_available


def test_torch_numpy_not_available_is_opposite_of_torch_numpy_available() -> None:
    numpy_available = fixtures.torch_numpy_available.args[0]
    numpy_not_available = fixtures.torch_numpy_not_available.args[0]
    assert numpy_available != numpy_not_available


##############################
#     Tests for requires     #
##############################


def test_requires_dependency_available() -> None:
    available, not_available = fixtures.requires(True, "NumPy")
    assert available.args[0] is False  # not skipped
    assert not_available.args[0] is True  # skipped


def test_requires_dependency_not_available() -> None:
    available, not_available = fixtures.requires(False, "NumPy")
    assert available.args[0] is True  # skipped
    assert not_available.args[0] is False  # not skipped


def test_requires_reasons() -> None:
    available, not_available = fixtures.requires(True, "NumPy")
    assert available.kwargs["reason"] == "Requires NumPy"
    assert not_available.kwargs["reason"] == "Skip if NumPy is available"


def test_requires_reasons_plural() -> None:
    available, not_available = fixtures.requires(True, "PyTorch and CUDA", plural=True)
    assert available.kwargs["reason"] == "Requires PyTorch and CUDA"
    assert not_available.kwargs["reason"] == "Skip if PyTorch and CUDA are available"


def test_requires_returns_skipif_marks() -> None:
    available, not_available = fixtures.requires(True, "NumPy")
    assert isinstance(available, pytest.MarkDecorator)
    assert isinstance(not_available, pytest.MarkDecorator)
    assert available.name == not_available.name == "skipif"


@pytest.mark.parametrize(
    "available_name",
    [
        "colorlog",
        "jax",
        "numpy",
        "packaging",
        "pandas",
        "polars",
        "pyarrow",
        "pydantic",
        "rich",
        "torch",
        "torch_cuda",
        "torch_mps",
        "torch_numpy",
        "xarray",
    ],
)
def test_available_and_not_available_marks_are_opposite(available_name: str) -> None:
    available = getattr(fixtures, f"{available_name}_available")
    not_available = getattr(fixtures, f"{available_name}_not_available")
    assert available.args[0] != not_available.args[0]
    assert available.kwargs["reason"].startswith("Requires ")
    assert not_available.kwargs["reason"].startswith("Skip if ")
    assert not_available.kwargs["reason"].endswith(" available")


def test_requires_is_public() -> None:
    assert "requires" in fixtures.__all__
    assert callable(fixtures.requires)


def test_all_names_are_requires_and_the_available_and_not_available_marks() -> None:
    marks = [name for name in fixtures.__all__ if name != "requires"]
    assert len(marks) == 28
    for name in marks:
        assert name.endswith(("_available", "_not_available"))
        assert isinstance(getattr(fixtures, name), pytest.MarkDecorator)

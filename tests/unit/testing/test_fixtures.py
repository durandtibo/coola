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

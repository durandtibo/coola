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
def test_available_fixture_has_not_available_counterpart_available(available_name: str) -> None:
    r"""Every ``*_available`` fixture must have a ``*_not_available``
    counterpart."""
    available = getattr(fixtures, f"{available_name}_available")
    getattr(fixtures, f"{available_name}_not_available")
    assert isinstance(available, pytest.MarkDecorator)


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
def test_available_fixture_has_not_available_counterpart_not_available(available_name: str) -> None:
    getattr(fixtures, f"{available_name}_available")
    not_available = getattr(fixtures, f"{available_name}_not_available")
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


def test_requires_dependency_available_available() -> None:
    available, _not_available = fixtures.requires(True, "NumPy")
    assert available.args[0] is False  # not skipped


def test_requires_dependency_available_not_available() -> None:
    _available, not_available = fixtures.requires(True, "NumPy")
    assert not_available.args[0] is True  # skipped


def test_requires_dependency_not_available_available() -> None:
    available, _not_available = fixtures.requires(False, "NumPy")
    assert available.args[0] is True  # skipped


def test_requires_dependency_not_available_not_available() -> None:
    _available, not_available = fixtures.requires(False, "NumPy")
    assert not_available.args[0] is False  # not skipped


def test_requires_reasons_available() -> None:
    available, _not_available = fixtures.requires(True, "NumPy")
    assert available.kwargs["reason"] == "Requires NumPy"


def test_requires_reasons_not_available() -> None:
    _available, not_available = fixtures.requires(True, "NumPy")
    assert not_available.kwargs["reason"] == "Skip if NumPy is available"


def test_requires_reasons_plural_available() -> None:
    available, _not_available = fixtures.requires(True, "PyTorch and CUDA", plural=True)
    assert available.kwargs["reason"] == "Requires PyTorch and CUDA"


def test_requires_reasons_plural_not_available() -> None:
    _available, not_available = fixtures.requires(True, "PyTorch and CUDA", plural=True)
    assert not_available.kwargs["reason"] == "Skip if PyTorch and CUDA are available"


def test_requires_returns_skipif_marks_available() -> None:
    available, _not_available = fixtures.requires(True, "NumPy")
    assert isinstance(available, pytest.MarkDecorator)


def test_requires_returns_skipif_marks_not_available() -> None:
    _available, not_available = fixtures.requires(True, "NumPy")
    assert isinstance(not_available, pytest.MarkDecorator)


def test_requires_returns_skipif_marks_name() -> None:
    available, not_available = fixtures.requires(True, "NumPy")
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
def test_available_and_not_available_marks_are_opposite_opposite_args(available_name: str) -> None:
    available = getattr(fixtures, f"{available_name}_available")
    not_available = getattr(fixtures, f"{available_name}_not_available")
    assert available.args[0] != not_available.args[0]


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
def test_available_and_not_available_marks_are_opposite_available_reason_prefix(
    available_name: str,
) -> None:
    available = getattr(fixtures, f"{available_name}_available")
    getattr(fixtures, f"{available_name}_not_available")
    assert available.kwargs["reason"].startswith("Requires ")


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
def test_available_and_not_available_marks_are_opposite_not_available_reason_prefix(
    available_name: str,
) -> None:
    getattr(fixtures, f"{available_name}_available")
    not_available = getattr(fixtures, f"{available_name}_not_available")
    assert not_available.kwargs["reason"].startswith("Skip if ")


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
def test_available_and_not_available_marks_are_opposite_not_available_reason_suffix(
    available_name: str,
) -> None:
    getattr(fixtures, f"{available_name}_available")
    not_available = getattr(fixtures, f"{available_name}_not_available")
    assert not_available.kwargs["reason"].endswith(" available")


def test_requires_is_in_all() -> None:
    assert "requires" in fixtures.__all__


def test_requires_is_callable() -> None:
    assert callable(fixtures.requires)


MARK_NAMES = [name for name in fixtures.__all__ if name != "requires"]


def test_all_names_are_requires_and_the_available_and_not_available_marks() -> None:
    assert len(MARK_NAMES) == 28


@pytest.mark.parametrize("name", MARK_NAMES)
def test_all_marks_are_available_or_not_available(name: str) -> None:
    assert name.endswith(("_available", "_not_available"))


@pytest.mark.parametrize("name", MARK_NAMES)
def test_all_marks_are_mark_decorators(name: str) -> None:
    assert isinstance(getattr(fixtures, name), pytest.MarkDecorator)


##########################################
#     Tests for the lazy attributes      #
##########################################


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'unknown_available'"):
        _ = fixtures.unknown_available


def test_unknown_attribute_close_to_a_marker_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="numpy_availabl"):
        _ = fixtures.numpy_availabl


def test_available_marker_is_cached() -> None:
    assert fixtures.pandas_available is fixtures.pandas_available


def test_not_available_marker_is_cached() -> None:
    assert fixtures.pandas_not_available is fixtures.pandas_not_available


def test_available_marker_is_created_with_the_not_available_marker() -> None:
    fixtures.polars_not_available  # noqa: B018
    assert "polars_available" in vars(fixtures)


def test_available_and_not_available_markers_are_created_together() -> None:
    available = fixtures.polars_not_available
    assert vars(fixtures)["polars_not_available"] is available


@pytest.mark.parametrize("name", MARK_NAMES)
def test_all_names_are_markers(name: str) -> None:
    assert isinstance(getattr(fixtures, name), pytest.MarkDecorator)

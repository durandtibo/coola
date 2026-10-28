from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.testing import fixtures

if TYPE_CHECKING:
    from collections.abc import Generator

KEYS = list(fixtures._DEPENDENCIES)


@pytest.fixture
def two_dependencies(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    r"""Replace the table by two cheap rows, and drop the marks that
    ``__getattr__`` caches in the module."""
    monkeypatch.setattr(
        fixtures,
        "_DEPENDENCIES",
        {
            "alpha": fixtures._Dependency(lambda: True, "Alpha"),
            "beta": fixtures._Dependency(lambda: False, "Beta with B"),
        },
    )
    yield
    for key in ("alpha", "beta"):
        vars(fixtures).pop(f"{key}_available", None)
        vars(fixtures).pop(f"{key}_not_available", None)


#############################
#     Tests for requires    #
#############################


def test_requires_available_does_not_skip_when_available() -> None:
    available, _ = fixtures.requires(True, "NumPy")
    assert available.args[0] is False


def test_requires_available_skips_when_not_available() -> None:
    available, _ = fixtures.requires(False, "NumPy")
    assert available.args[0] is True


def test_requires_not_available_skips_when_available() -> None:
    _, not_available = fixtures.requires(True, "NumPy")
    assert not_available.args[0] is True


def test_requires_not_available_does_not_skip_when_not_available() -> None:
    _, not_available = fixtures.requires(False, "NumPy")
    assert not_available.args[0] is False


def test_requires_reasons_available() -> None:
    available, _ = fixtures.requires(True, "NumPy")
    assert available.kwargs["reason"] == "Requires NumPy"


def test_requires_reasons_not_available() -> None:
    _, not_available = fixtures.requires(True, "NumPy")
    assert not_available.kwargs["reason"] == "Skip if NumPy is available"


def test_requires_returns_skipif_marks() -> None:
    available, not_available = fixtures.requires(True, "NumPy")
    assert available.name == not_available.name == "skipif"


def test_requires_rejects_plural() -> None:
    with pytest.raises(TypeError, match="plural"):
        fixtures.requires(True, "NumPy", plural=True)  # ty: ignore[unknown-argument]


######################################
#     Tests for the table of marks   #
######################################


@pytest.fixture
def table_marks(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[pytest.MarkDecorator, pytest.MarkDecorator]:
    r"""Build the pair of a real key from a patched check, so the test
    does not import the library."""
    key = request.param
    monkeypatch.setitem(fixtures._DEPENDENCIES, key, fixtures._Dependency(lambda: True, "X"))
    for suffix in ("_available", "_not_available"):
        monkeypatch.delitem(vars(fixtures), f"{key}{suffix}", raising=False)
    marks = getattr(fixtures, f"{key}_available"), getattr(fixtures, f"{key}_not_available")
    for suffix in ("_available", "_not_available"):
        vars(fixtures).pop(f"{key}{suffix}", None)
    return marks


@pytest.mark.parametrize("table_marks", KEYS, indirect=True)
def test_every_key_gives_two_skipif_marks(
    table_marks: tuple[pytest.MarkDecorator, pytest.MarkDecorator],
) -> None:
    assert [mark.name for mark in table_marks] == ["skipif", "skipif"]


@pytest.mark.parametrize("table_marks", KEYS, indirect=True)
def test_every_key_gives_opposite_marks(
    table_marks: tuple[pytest.MarkDecorator, pytest.MarkDecorator],
) -> None:
    assert table_marks[0].args[0] != table_marks[1].args[0]


@pytest.mark.parametrize("table_marks", KEYS, indirect=True)
def test_every_key_available_reason_prefix(
    table_marks: tuple[pytest.MarkDecorator, pytest.MarkDecorator],
) -> None:
    assert table_marks[0].kwargs["reason"].startswith("Requires ")


@pytest.mark.parametrize("table_marks", KEYS, indirect=True)
def test_every_key_not_available_reason_prefix(
    table_marks: tuple[pytest.MarkDecorator, pytest.MarkDecorator],
) -> None:
    assert table_marks[1].kwargs["reason"].startswith("Skip if ")


@pytest.mark.parametrize("table_marks", KEYS, indirect=True)
def test_every_key_not_available_reason_suffix(
    table_marks: tuple[pytest.MarkDecorator, pytest.MarkDecorator],
) -> None:
    assert table_marks[1].kwargs["reason"].endswith(" is available")


@pytest.mark.parametrize(
    ("key", "name"),
    [
        ("torch_cuda", "PyTorch with CUDA"),
        ("torch_mps", "PyTorch with MPS"),
        ("torch_numpy", "PyTorch with NumPy"),
    ],
)
def test_composite_names_are_singular(key: str, name: str) -> None:
    assert fixtures._DEPENDENCIES[key].name == name


##################################
#     Tests for the attributes   #
##################################


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'unknown_available'"):
        _ = fixtures.unknown_available


def test_unknown_attribute_lists_the_valid_names() -> None:
    with pytest.raises(AttributeError, match="the valid names are: requires, colorlog_available"):
        _ = fixtures.unknown_available


def test_unknown_key_with_a_valid_suffix_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="numpyy_available"):
        _ = fixtures.numpyy_available


def test_suffix_alone_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute '_available'"):
        _ = fixtures._available


@pytest.mark.usefixtures("two_dependencies")
def test_available_mark_is_cached() -> None:
    assert fixtures.alpha_available is fixtures.alpha_available


@pytest.mark.usefixtures("two_dependencies")
def test_marks_are_created_together() -> None:
    fixtures.alpha_not_available  # noqa: B018
    assert "alpha_available" in vars(fixtures)


@pytest.mark.usefixtures("two_dependencies")
def test_check_runs_on_first_access() -> None:
    assert fixtures.beta_available.args[0] is True


def test_all_starts_with_requires_and_the_first_pair() -> None:
    assert fixtures.__all__[:3] == ["requires", "colorlog_available", "colorlog_not_available"]


def test_all_has_two_names_per_row_and_requires() -> None:
    assert len(fixtures.__all__) == 2 * len(fixtures._DEPENDENCIES) + 1


def test_dir_is_sorted_all() -> None:
    assert dir(fixtures) == sorted(fixtures.__all__)


def test_cuda_check_uses_coola_utils_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fixtures, "import_module", lambda _: Mock(is_cuda_available=lambda: True))
    assert fixtures._is_cuda_available() is True


def test_mps_check_uses_coola_utils_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fixtures, "import_module", lambda _: Mock(is_mps_available=lambda: False))
    assert fixtures._is_mps_available() is False

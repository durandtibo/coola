from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from coola.testing import marks

if TYPE_CHECKING:
    from collections.abc import Generator

KEYS = list(marks._DEPENDENCIES)


@pytest.fixture
def two_dependencies(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    r"""Replace the table by two cheap rows, and drop the marks that
    ``__getattr__`` caches in the module."""
    monkeypatch.setattr(
        marks,
        "_DEPENDENCIES",
        {
            "alpha": marks._Dependency(lambda: True, "Alpha"),
            "beta": marks._Dependency(lambda: False, "Beta with B"),
        },
    )
    yield
    for key in ("alpha", "beta"):
        vars(marks).pop(f"{key}_available", None)
        vars(marks).pop(f"{key}_not_available", None)


#############################
#     Tests for requires    #
#############################


def test_requires_available_does_not_skip_when_available() -> None:
    available, _ = marks.requires(True, "NumPy")
    assert available.args[0] is False


def test_requires_available_skips_when_not_available() -> None:
    available, _ = marks.requires(False, "NumPy")
    assert available.args[0] is True


def test_requires_not_available_skips_when_available() -> None:
    _, not_available = marks.requires(True, "NumPy")
    assert not_available.args[0] is True


def test_requires_not_available_does_not_skip_when_not_available() -> None:
    _, not_available = marks.requires(False, "NumPy")
    assert not_available.args[0] is False


def test_requires_reasons_available() -> None:
    available, _ = marks.requires(True, "NumPy")
    assert available.kwargs["reason"] == "Requires NumPy"


def test_requires_reasons_not_available() -> None:
    _, not_available = marks.requires(True, "NumPy")
    assert not_available.kwargs["reason"] == "Skip if NumPy is available"


def test_requires_returns_skipif_marks() -> None:
    available, not_available = marks.requires(True, "NumPy")
    assert available.name == not_available.name == "skipif"


def test_requires_rejects_plural() -> None:
    with pytest.raises(TypeError, match="plural"):
        marks.requires(True, "NumPy", plural=True)  # ty: ignore[unknown-argument]


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
    monkeypatch.setitem(marks._DEPENDENCIES, key, marks._Dependency(lambda: True, "X"))
    for suffix in ("_available", "_not_available"):
        monkeypatch.delitem(vars(marks), f"{key}{suffix}", raising=False)
    pair = getattr(marks, f"{key}_available"), getattr(marks, f"{key}_not_available")
    for suffix in ("_available", "_not_available"):
        vars(marks).pop(f"{key}{suffix}", None)
    return pair


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
    assert marks._DEPENDENCIES[key].name == name


##################################
#     Tests for the attributes   #
##################################


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'unknown_available'"):
        _ = marks.unknown_available


def test_unknown_attribute_lists_the_valid_names() -> None:
    with pytest.raises(AttributeError, match="the valid names are: requires, colorlog_available"):
        _ = marks.unknown_available


def test_unknown_key_with_a_valid_suffix_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="numpyy_available"):
        _ = marks.numpyy_available


def test_suffix_alone_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute '_available'"):
        _ = marks._available


@pytest.mark.usefixtures("two_dependencies")
def test_available_mark_is_cached() -> None:
    assert marks.alpha_available is marks.alpha_available


@pytest.mark.usefixtures("two_dependencies")
def test_marks_are_created_together() -> None:
    _ = marks.alpha_not_available
    assert "alpha_available" in vars(marks)


@pytest.mark.usefixtures("two_dependencies")
def test_check_runs_on_first_access() -> None:
    assert marks.beta_available.args[0] is True


def test_all_starts_with_requires_and_the_first_pair() -> None:
    assert marks.__all__[:3] == ["requires", "colorlog_available", "colorlog_not_available"]


def test_all_has_two_names_per_row_and_requires() -> None:
    assert len(marks.__all__) == 2 * len(marks._DEPENDENCIES) + 1


def test_dir_is_sorted_all() -> None:
    assert dir(marks) == sorted(marks.__all__)


def test_cuda_check_uses_coola_utils_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(marks, "import_module", lambda _: Mock(is_cuda_available=lambda: True))
    assert marks._is_cuda_available() is True


def test_mps_check_uses_coola_utils_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(marks, "import_module", lambda _: Mock(is_mps_available=lambda: False))
    assert marks._is_mps_available() is False

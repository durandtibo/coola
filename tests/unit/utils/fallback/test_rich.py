from __future__ import annotations

import pytest

from coola.utils.fallback import rich
from coola.utils.imports import MissingPackageError

FAKE_CLASSES = [
    "BarColumn",
    "Markdown",
    "MofNCompleteColumn",
    "Panel",
    "Pretty",
    "Progress",
    "RichHandler",
    "SpinnerColumn",
    "TaskProgressColumn",
    "TextColumn",
    "TimeElapsedColumn",
    "TimeRemainingColumn",
]


@pytest.mark.parametrize("name", FAKE_CLASSES)
def test_fake_class_is_class(name: str) -> None:
    assert isinstance(getattr(rich, name), type)


@pytest.mark.parametrize("name", FAKE_CLASSES)
def test_fake_class_instantiation(name: str) -> None:
    with pytest.raises(MissingPackageError, match=r"'rich' package is required but not installed."):
        getattr(rich, name)()


def test_fake_class_instantiation_with_args() -> None:
    with pytest.raises(MissingPackageError, match=r"'rich' package is required but not installed."):
        rich.Pretty({"a": 1}, indent_size=2)


def test_get_console() -> None:
    with pytest.raises(MissingPackageError, match=r"'rich' package is required but not installed."):
        rich.get_console()


@pytest.mark.parametrize("name", ["MINIMAL", "ROUNDED"])
def test_box_is_none(name: str) -> None:
    assert getattr(rich, name) is None

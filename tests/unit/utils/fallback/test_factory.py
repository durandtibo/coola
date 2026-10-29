from __future__ import annotations

import pytest

from coola.utils.fallback.factory import make_fake_class, make_fake_function
from coola.utils.imports import MissingPackageError

MESSAGE = (
    "'missing_pkg' package is required but not installed. "
    "You can install 'missing_pkg' package with the command:\n\n"
    "pip install missing_pkg\n\nor\n\n"
    "uv pip install missing_pkg\n"
)


def test_make_fake_class_returns_type() -> None:
    assert isinstance(make_fake_class("missing_pkg"), type)


def test_make_fake_class_name() -> None:
    assert make_fake_class("missing_pkg").__name__ == "FakeClass"


def test_make_fake_class_instantiation_raises_error() -> None:
    fake_class = make_fake_class("missing_pkg")
    with pytest.raises(MissingPackageError, match="'missing_pkg' package is required") as info:
        fake_class()
    assert str(info.value) == MESSAGE


def test_make_fake_class_instantiation_raises_error_with_args() -> None:
    fake_class = make_fake_class("missing_pkg")
    with pytest.raises(MissingPackageError, match=r"'missing_pkg' package is required"):
        fake_class(1, 2, key="value")


def test_make_fake_class_error_name() -> None:
    fake_class = make_fake_class("missing_pkg")
    with pytest.raises(MissingPackageError, match="'missing_pkg' package is required") as info:
        fake_class()
    assert info.value.name == "missing_pkg"


def test_make_fake_class_independent_instances() -> None:
    fake_class1 = make_fake_class("missing_pkg")
    fake_class2 = make_fake_class("missing_pkg")
    assert fake_class1 is not fake_class2


def test_make_fake_function_returns_callable() -> None:
    assert callable(make_fake_function("missing_pkg"))


def test_make_fake_function_call_raises_error() -> None:
    fake_function = make_fake_function("missing_pkg")
    with pytest.raises(MissingPackageError, match="'missing_pkg' package is required") as info:
        fake_function()
    assert str(info.value) == MESSAGE


def test_make_fake_function_call_raises_error_with_args() -> None:
    fake_function = make_fake_function("missing_pkg")
    with pytest.raises(MissingPackageError, match=r"'missing_pkg' package is required"):
        fake_function(1, 2, key="value")

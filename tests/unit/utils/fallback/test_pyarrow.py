from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.pyarrow import pyarrow
from coola.utils.imports import MissingPackageError


def test_pyarrow_is_module_type() -> None:
    assert isinstance(pyarrow, ModuleType)


def test_pyarrow_module_name() -> None:
    assert pyarrow.__name__ == "pyarrow"


def test_pyarrow_array_class_exists() -> None:
    assert hasattr(pyarrow, "Array")


def test_pyarrow_array_is_class() -> None:
    assert isinstance(pyarrow.Array, type)


def test_pyarrow_array_instantiation() -> None:
    with pytest.raises(
        MissingPackageError, match=r"'pyarrow' package is required but not installed."
    ):
        pyarrow.Array()


def test_pyarrow_array_instantiation_with_args() -> None:
    with pytest.raises(
        MissingPackageError, match=r"'pyarrow' package is required but not installed."
    ):
        pyarrow.Array([1, 2, 3])


@pytest.mark.parametrize("name", ["ChunkedArray", "Table"])
def test_pyarrow_fake_class_exists(name: str) -> None:
    assert isinstance(getattr(pyarrow, name), type)


@pytest.mark.parametrize("name", ["ChunkedArray", "Table"])
def test_pyarrow_fake_class_instantiation_raises(name: str) -> None:
    with pytest.raises(MissingPackageError, match="'pyarrow' package is required"):
        getattr(pyarrow, name)()

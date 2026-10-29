from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.scipy import scipy
from coola.utils.imports import MissingPackageError


def test_scipy_is_module_type() -> None:
    assert isinstance(scipy, ModuleType)


def test_scipy_module_name() -> None:
    assert scipy.__name__ == "scipy"


def test_scipy_sparse_spmatrix_is_class() -> None:
    assert isinstance(scipy.sparse.spmatrix, type)


def test_scipy_sparse_sparray_is_class() -> None:
    assert isinstance(scipy.sparse.sparray, type)


def test_scipy_sparse_spmatrix_instantiation_raises() -> None:
    with pytest.raises(MissingPackageError, match=r"'scipy' package is required"):
        scipy.sparse.spmatrix()

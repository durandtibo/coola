from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.jax import jax


def test_jax_is_module_type() -> None:
    assert isinstance(jax, ModuleType)


def test_jax_module_name() -> None:
    assert jax.__name__ == "jax"


def test_jax_nested_module_access() -> None:
    assert hasattr(jax, "numpy")


def test_jax_ndarray_exists() -> None:
    assert hasattr(jax.numpy, "ndarray")


def test_jax_ndarray_is_class() -> None:
    assert isinstance(jax.numpy.ndarray, type)


def test_jax_ndarray_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'jax' package is required but not installed."):
        jax.numpy.ndarray()

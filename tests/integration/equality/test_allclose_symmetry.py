from __future__ import annotations

import pytest

from coola.equality import objects_are_allclose
from coola.testing.fixtures import jax_available, numpy_available, torch_available
from coola.utils.imports import is_jax_available, is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
if is_torch_available():
    import torch
if is_jax_available():
    import jax.numpy as jnp

KWARGS = {"rtol": 0.01, "atol": 0.0}


def test_objects_are_allclose_scalar_is_symmetric() -> None:
    assert objects_are_allclose(100.0, 101.005, **KWARGS) == objects_are_allclose(
        101.005, 100.0, **KWARGS
    )


@numpy_available
@pytest.mark.parametrize(("actual", "expected"), [(100.0, 101.005), (101.005, 100.0)])
def test_objects_are_allclose_numpy_is_symmetric(actual: float, expected: float) -> None:
    assert objects_are_allclose(np.array([actual]), np.array([expected]), **KWARGS)


@torch_available
@pytest.mark.parametrize(("actual", "expected"), [(100.0, 101.005), (101.005, 100.0)])
def test_objects_are_allclose_torch_is_symmetric(actual: float, expected: float) -> None:
    assert objects_are_allclose(torch.tensor([actual]), torch.tensor([expected]), **KWARGS)


@jax_available
@pytest.mark.parametrize(("actual", "expected"), [(100.0, 101.005), (101.005, 100.0)])
def test_objects_are_allclose_jax_is_symmetric(actual: float, expected: float) -> None:
    assert objects_are_allclose(jnp.array([actual]), jnp.array([expected]), **KWARGS)


@numpy_available
def test_objects_are_allclose_numpy_nested_in_a_dict_is_symmetric() -> None:
    assert objects_are_allclose({"a": np.array([101.005])}, {"a": np.array([100.0])}, **KWARGS)

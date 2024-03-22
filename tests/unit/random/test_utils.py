from __future__ import annotations

from coola import objects_are_equal
from coola.random import (
    NumpyRandomManager,
    RandomRandomManager,
    TorchRandomManager,
    get_random_managers,
)
from coola.testing import numpy_available, torch_available

#########################################
#     Tests for get_random_managers     #
#########################################


def test_get_random_managers() -> None:
    mapping = get_random_managers()
    assert len(mapping) >= 1
    assert objects_are_equal(mapping["random"], RandomRandomManager())


@numpy_available
def test_get_random_managers_numpy() -> None:
    mapping = get_random_managers()
    assert len(mapping) >= 1
    assert objects_are_equal(mapping["numpy"], NumpyRandomManager())


@torch_available
def test_get_random_managers_torch() -> None:
    mapping = get_random_managers()
    assert len(mapping) >= 1
    assert objects_are_equal(mapping["torch"], TorchRandomManager())

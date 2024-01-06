from __future__ import annotations

from coola.equality.comparators import (
    ArrayEqualityComparator,
    DefaultEqualityComparator,
    TensorEqualityComparator,
    get_type_comparator_mapping,
)
from coola.testing import torch_available
from coola.utils.imports import is_numpy_available, is_torch_available, numpy_available

if is_numpy_available():
    import numpy as np

if is_torch_available():
    import torch

#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


def test_get_type_comparator_mapping() -> None:
    mapping = get_type_comparator_mapping()
    assert len(mapping) >= 1
    assert isinstance(mapping[object], DefaultEqualityComparator)


@numpy_available
def test_get_type_comparator_mapping_numpy() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[np.ndarray], ArrayEqualityComparator)


@torch_available
def test_get_type_comparator_mapping_torch() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[torch.Tensor], TensorEqualityComparator)

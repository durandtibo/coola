from __future__ import annotations

from collections.abc import Mapping, Sequence

from coola.equality.comparators import (
    DefaultEqualityComparator,
    MappingEqualityComparator,
    NumpyArrayEqualityComparator,
    SequenceEqualityComparator,
    TorchTensorEqualityComparator,
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
    assert len(mapping) >= 6
    assert isinstance(mapping[Mapping], MappingEqualityComparator)
    assert isinstance(mapping[Sequence], SequenceEqualityComparator)
    assert isinstance(mapping[dict], MappingEqualityComparator)
    assert isinstance(mapping[list], SequenceEqualityComparator)
    assert isinstance(mapping[object], DefaultEqualityComparator)
    assert isinstance(mapping[tuple], SequenceEqualityComparator)


@numpy_available
def test_get_type_comparator_mapping_numpy() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[np.ndarray], NumpyArrayEqualityComparator)


@torch_available
def test_get_type_comparator_mapping_torch() -> None:
    mapping = get_type_comparator_mapping()
    assert isinstance(mapping[torch.Tensor], TorchTensorEqualityComparator)

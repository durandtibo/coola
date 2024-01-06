from __future__ import annotations

from coola.equality.comparators import (
    ArrayEqualityComparator,
    DefaultEqualityComparator,
    get_type_comparator_mapping,
)
from coola.utils.imports import is_numpy_available, numpy_available

if is_numpy_available():
    import numpy as np


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

from __future__ import annotations

import pytest

from coola.recursive import DefaultTransformer, SequenceTransformer
from tests.unit.recursive.helpers import expand_not_none

#####################################
#     Tests for expand_not_none     #
#####################################


def test_expand_not_none_returns_the_children() -> None:
    children, _ = expand_not_none(SequenceTransformer(), [1, 2])
    assert children == [1, 2]


def test_expand_not_none_returns_the_rebuild_function() -> None:
    _, rebuild = expand_not_none(SequenceTransformer(), (1, 2))
    assert rebuild([3, 4]) == (3, 4)


def test_expand_not_none_leaf_raises() -> None:
    with pytest.raises(AssertionError, match="treats 1 as a leaf"):
        expand_not_none(DefaultTransformer(), 1)

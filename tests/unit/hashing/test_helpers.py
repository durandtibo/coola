from __future__ import annotations

import pytest

from coola.hashing import Node, ReprHasher, SequenceHasher
from tests.unit.hashing.helpers import encode_leaf, encode_node

#################################
#     Tests for encode_leaf     #
#################################


def test_encode_leaf_returns_the_encoding() -> None:
    assert encode_leaf(ReprHasher(), 1) == b"int\x001"


def test_encode_leaf_node_raises() -> None:
    with pytest.raises(AssertionError, match="as a node"):
        encode_leaf(SequenceHasher(), [1])


#################################
#     Tests for encode_node     #
#################################


def test_encode_node_returns_the_node() -> None:
    assert isinstance(encode_node(SequenceHasher(), [1]), Node)


def test_encode_node_leaf_raises() -> None:
    with pytest.raises(AssertionError, match="as a leaf"):
        encode_node(ReprHasher(), 1)

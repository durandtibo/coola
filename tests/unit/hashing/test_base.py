from __future__ import annotations

import hashlib

import pytest

from coola.hashing import BaseHasher, Node, digest, tagged

#########################
#     Tests for Node     #
#########################


def test_node_defaults() -> None:
    node = Node("list", [1, 2])
    assert (node.tag, node.children, node.ordered, node.group) == ("list", [1, 2], True, 1)


def test_node_repr() -> None:
    assert repr(Node("set", {1}, ordered=False)) == (
        "Node(tag='set', children={1}, ordered=False, group=1)"
    )


def test_node_group() -> None:
    assert Node("dict", [], ordered=False, group=2).group == 2


###########################
#     Tests for tagged     #
###########################


def test_tagged_text_payload() -> None:
    assert tagged("int", "1") == b"int\x001"


def test_tagged_bytes_payload() -> None:
    assert tagged("bytes", b"\x01\x02") == b"bytes\x00\x01\x02"


def test_tagged_without_payload() -> None:
    assert tagged("NA") == b"NA\x00"


def test_tagged_text_is_encoded_in_utf8() -> None:
    assert tagged("str", "é") == b"str\x00\xc3\xa9"


def test_tagged_lone_surrogate_is_kept() -> None:
    assert tagged("str", "\ud800") == b"str\x00\xed\xa0\x80"


def test_tagged_lone_surrogate_differs_from_a_valid_string() -> None:
    assert tagged("str", "\ud800") != tagged("str", "�")


def test_tagged_different_tags_differ() -> None:
    assert tagged("int", "1") != tagged("str", "1")


###########################
#     Tests for digest     #
###########################


def test_digest_length() -> None:
    assert len(digest(b"abc")) == 64


def test_digest_is_the_blake2b_with_the_version_of_the_format() -> None:
    expected = hashlib.blake2b(b"abc", digest_size=64, person=b"coola-hash-1").digest()
    assert digest(b"abc") == expected


def test_digest_differs_from_the_blake2b_without_the_version() -> None:
    assert digest(b"abc") != hashlib.blake2b(b"abc", digest_size=64).digest()


def test_digest_is_deterministic() -> None:
    assert digest(b"abc") == digest(b"abc")


def test_digest_different_inputs() -> None:
    assert digest(b"abc") != digest(b"abd")


def test_digest_empty() -> None:
    assert len(digest(b"")) == 64


####################################
#     Tests for BaseHasher     #
####################################


def test_base_hasher_cannot_be_instantiated_without_encode() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseHasher.__new__(BaseHasher)

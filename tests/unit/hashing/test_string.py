from __future__ import annotations

from collections import UserString
from enum import Enum

import pytest

from coola.hashing import HasherRegistry, StringHasher, tagged
from tests.unit.hashing.helpers import hash_with, leaf_hash


class MixedStr(str, Enum):  # noqa: UP042
    A = "a"


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: StringHasher()})


##################################
#     Tests for StringHasher     #
##################################


def test_string_hasher_repr() -> None:
    assert repr(StringHasher()) == "StringHasher()"


def test_string_hasher_str() -> None:
    assert str(StringHasher()) == "StringHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            "hello",
            "2998e64e4b20ac91774f304f5527a5160d30249cdcdc984effac931cf54196e0",
            id="lowercase",
        ),
        pytest.param(
            "Hello",
            "d2e6b55494e319a95d4724ef225b9c8fa1a8cecae70f33f9b4421bfc1703dbee",
            id="uppercase_first",
        ),
        pytest.param(
            "hello ",
            "58629ea5bbc011db083e224e27e7dcc4b88c747964447d3fb51ab344fcc42a89",
            id="trailing_space",
        ),
        pytest.param(
            "",
            "13794f6a993bed50b22118af38aaa84e510b29e67ee82abb91bc758b9b8ba4d3",
            id="empty_string",
        ),
        pytest.param(
            "Meowwwwww",
            "dc79d6439fb717960d695a964b67a3c46f68e692b07cc1b9e59828d6a9299f10",
            id="meow",
        ),
    ],
)
def test_string_hasher_hash_parametrized(
    data: str, expected: str, registry: HasherRegistry
) -> None:
    assert hash_with(StringHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "2998e64e4b20ac91", id="16"),
        pytest.param(32, "2998e64e4b20ac91774f304f5527a516", id="32"),
        pytest.param(
            64, "2998e64e4b20ac91774f304f5527a5160d30249cdcdc984effac931cf54196e0", id="64-default"
        ),
    ],
)
def test_string_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(StringHasher(), "hello", registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_string_hasher_hash_length_length_is_respected(
    length: int, registry: HasherRegistry
) -> None:
    result = hash_with(StringHasher(), "hello", registry, length=length)
    assert len(result) == length


def test_string_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(StringHasher(), "hello", registry), str)


def test_string_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(hasher, "hello", registry)


def test_string_hasher_hash_is_case_sensitive(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) != hash_with(hasher, "Hello", registry)


def test_string_hasher_hash_is_whitespace_sensitive(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) != hash_with(hasher, "hello ", registry)


def test_string_hasher_hash_different_strings_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) != hash_with(hasher, "world", registry)


def test_string_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # StringHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(
        hasher, "hello", registry, ignore_unhashable=True
    )


def test_string_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(hasher, "hello", empty_registry)


# --- Determinism ---


# --- Sensitivity ---


# --- Invalid length ---


def test_string_hasher_hash_is_tagged() -> None:
    assert hash_with(StringHasher(), "hello", HasherRegistry()) == leaf_hash(tagged("str", "hello"))


##########################################
#     Tests for validate_hash_length     #
##########################################


def test_string_hasher_hash_user_string_differs_from_str(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, UserString("abc"), registry) != hash_with(hasher, "abc", registry)


def test_string_hasher_hash_user_string_equals_the_tagged_text(registry: HasherRegistry) -> None:
    assert hash_with(StringHasher(), UserString("abc"), registry) == leaf_hash(
        tagged("UserString", "abc")
    )


def test_string_hasher_encode_valid_utf8() -> None:
    assert StringHasher().encode("é") == b"str\x00\xc3\xa9"


def test_string_hasher_encode_lone_surrogate_differs_from_the_replacement_character() -> None:
    assert StringHasher().encode("\ud800") != StringHasher().encode("\ufffd")


def test_string_hasher_encode_lone_surrogates_differ() -> None:
    assert StringHasher().encode("\ud800") != StringHasher().encode("\ud801")

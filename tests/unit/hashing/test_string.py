from __future__ import annotations

from collections import UserString
from enum import Enum

import pytest

from coola.hashing import HasherRegistry, StringHasher, hash_string, tagged
from coola.hashing.string import validate_hash_length
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


#################################
#     Tests for hash_string     #
#################################

# --- Return type and format ---


def test_hash_string_returns_str() -> None:
    assert isinstance(hash_string("hello"), str)


def test_hash_string_returns_lowercase_hex() -> None:
    assert all(c in "0123456789abcdef" for c in hash_string("hello"))


def test_hash_string_default_length_is_64() -> None:
    assert len(hash_string("hello")) == 64


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(2, id="min"),
        pytest.param(16, id="16"),
        pytest.param(32, id="32"),
        pytest.param(64, id="64-default"),
        pytest.param(128, id="max"),
    ],
)
def test_hash_string_output_length_matches_requested(length: int) -> None:
    assert len(hash_string("hello", length=length)) == length


# --- Determinism ---


def test_hash_string_is_deterministic() -> None:
    assert hash_string("hello") == hash_string("hello")


def test_hash_string_empty_string_is_valid_returns_str() -> None:
    result = hash_string("")
    assert isinstance(result, str)


def test_hash_string_empty_string_is_valid_default_length() -> None:
    result = hash_string("")
    assert len(result) == 64


# --- Sensitivity ---


def test_hash_string_different_inputs_produce_different_hashes() -> None:
    assert hash_string("hello") != hash_string("world")


def test_hash_string_same_prefix_different_inputs_produce_different_hashes() -> None:
    assert hash_string("hello1") != hash_string("hello2")


def test_hash_string_is_case_sensitive() -> None:
    assert hash_string("hello") != hash_string("Hello")


def test_hash_string_is_sensitive_to_whitespace() -> None:
    assert hash_string("hello") != hash_string("hello ")


# --- Invalid length ---


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(0, id="zero"),
        pytest.param(1, id="odd-below-min"),
        pytest.param(3, id="odd-small"),
        pytest.param(63, id="odd-near-max"),
        pytest.param(130, id="above-max"),
        pytest.param(-2, id="negative-even"),
        pytest.param(-1, id="negative-odd"),
    ],
)
def test_hash_string_raises_for_invalid_length(length: int) -> None:
    with pytest.raises(ValueError, match=str(abs(length))):
        hash_string("hello", length=length)


def test_string_hasher_hash_is_tagged() -> None:
    assert hash_with(StringHasher(), "hello", HasherRegistry()) == leaf_hash(tagged("str", "hello"))


def test_string_hasher_hash_differs_from_untagged_hash_string(registry: HasherRegistry) -> None:
    assert hash_with(StringHasher(), "hello", registry) != hash_string("hello")


def test_hash_string_lone_surrogate() -> None:
    assert len(hash_string("a\udc80")) == 64


def test_hash_string_lone_surrogate_differs_from_other_surrogate() -> None:
    assert hash_string("\udc80") != hash_string("\udc81")


def test_hash_string_lone_surrogate_differs_from_replacement_character() -> None:
    assert hash_string("\udc80") != hash_string("\ufffd")


def test_hash_string_unchanged_for_valid_utf8() -> None:
    assert hash_string("é", length=16) == "cb1abf8beff3192f"


##########################################
#     Tests for validate_hash_length     #
##########################################


@pytest.mark.parametrize("length", [2, 4, 64, 126, 128])
def test_validate_hash_length_valid(length: int) -> None:
    validate_hash_length(length)


@pytest.mark.parametrize("length", [0, 1, 3, 63, 127, 129, 130, -2, -1])
def test_validate_hash_length_invalid_raises(length: int) -> None:
    with pytest.raises(ValueError, match="length must be an even number between 2 and 128"):
        validate_hash_length(length)


def test_validate_hash_length_error_message_contains_length() -> None:
    with pytest.raises(ValueError, match="got 3"):
        validate_hash_length(3)


def test_validate_hash_length_returns_none() -> None:
    assert validate_hash_length(64) is None


def test_string_hasher_hash_user_string_differs_from_str(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, UserString("abc"), registry) != hash_with(hasher, "abc", registry)


def test_string_hasher_hash_user_string_equals_the_tagged_text(registry: HasherRegistry) -> None:
    assert hash_with(StringHasher(), UserString("abc"), registry) == leaf_hash(
        tagged("UserString", "abc")
    )

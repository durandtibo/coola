from __future__ import annotations

import pytest

from coola.hashing import hash_string

################################
#     Tests for hash_string    #
################################

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


def test_hash_string_empty_string_is_valid() -> None:
    result = hash_string("")
    assert isinstance(result, str)
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

from __future__ import annotations

import pytest

from coola.hashing import HasherRegistry, StringHasher, hash_string, hash_tagged


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: StringHasher()})


#################################
#     Tests for StringHasher    #
#################################


def test_string_hasher_repr() -> None:
    assert repr(StringHasher()) == "StringHasher()"


def test_string_hasher_str() -> None:
    assert str(StringHasher()) == "StringHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            "hello",
            "9d4bd2af5629d5a22405d7fe02fe410ee11ae44178b4ee0eb9ca4d7795ac9906",
            id="lowercase",
        ),
        pytest.param(
            "Hello",
            "6740b969578b13957189f26bdd86899a70899d4ff0ccff0e97ca1d3f8f411800",
            id="uppercase_first",
        ),
        pytest.param(
            "hello ",
            "9b059b06a9ab65ac68ea58aae9f0afe24e713169770d9470946d86c3ae4d78ec",
            id="trailing_space",
        ),
        pytest.param(
            "",
            "a024db9f50066fd82a7a075b9ed57faf248c835b586a8931b8f6ef2fb826cc29",
            id="empty_string",
        ),
        pytest.param(
            "Meowwwwww",
            "465fcb65cce407b13615e2324b10c99d6b94f7cca9cb69120d90415fb3632d9c",
            id="meow",
        ),
    ],
)
def test_string_hasher_hash_parametrized(
    data: str, expected: str, registry: HasherRegistry
) -> None:
    assert StringHasher().hash(data, registry=registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "8145498007a9ef19", id="16"),
        pytest.param(32, "2a34b1c3773ac9d514f55240b5b8a87e", id="32"),
        pytest.param(
            64, "9d4bd2af5629d5a22405d7fe02fe410ee11ae44178b4ee0eb9ca4d7795ac9906", id="64-default"
        ),
    ],
)
def test_string_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = StringHasher().hash("hello", registry=registry, length=length)
    assert result == expected
    assert len(result) == length


def test_string_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(StringHasher().hash("hello", registry=registry), str)


def test_string_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hasher.hash("hello", registry=registry) == hasher.hash("hello", registry=registry)


def test_string_hasher_hash_is_case_sensitive(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hasher.hash("hello", registry=registry) != hasher.hash("Hello", registry=registry)


def test_string_hasher_hash_is_whitespace_sensitive(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hasher.hash("hello", registry=registry) != hasher.hash("hello ", registry=registry)


def test_string_hasher_hash_different_strings_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = StringHasher()
    assert hasher.hash("hello", registry=registry) != hasher.hash("world", registry=registry)


def test_string_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # StringHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = StringHasher()
    assert hasher.hash("hello", registry=registry) == hasher.hash(
        "hello", registry=registry, ignore_unhashable=True
    )


def test_string_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = StringHasher()
    assert hasher.hash("hello", registry=registry) == hasher.hash("hello", registry=empty_registry)


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


def test_string_hasher_hash_is_tagged() -> None:
    assert StringHasher().hash("hello", registry=HasherRegistry()) == hash_tagged("str", "hello")


def test_string_hasher_hash_differs_from_untagged_hash_string(registry: HasherRegistry) -> None:
    assert StringHasher().hash("hello", registry=registry) != hash_string("hello")


def test_hash_string_lone_surrogate_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not valid UTF-8"):
        hash_string("\ud800")

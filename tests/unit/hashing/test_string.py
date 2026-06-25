from __future__ import annotations

import pytest

from coola.hashing import HasherRegistry, StringHasher, hash_string


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
            "324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf",
            id="lowercase",
        ),
        pytest.param(
            "Hello",
            "8b7ca7d27d9fc55fa30abfe515b3afb24e3fe89fdd02e2ac92bca2c96680642e",
            id="uppercase_first",
        ),
        pytest.param(
            "hello ",
            "5c945bdc1b01abc46c3bee5681b8bd080157aa9e76a5a39c575d856aab41d5cf",
            id="trailing_space",
        ),
        pytest.param(
            "",
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8",
            id="empty_string",
        ),
        pytest.param(
            "Meowwwwww",
            "1b06bfa9e842b52eaf47386798687ccd22697ed0198cfda4e0eee7e4650595f5",
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
        pytest.param(16, "a7b6eda801e5347d", id="16"),
        pytest.param(32, "46fb7408d4f285228f4af516ea25851b", id="32"),
        pytest.param(
            64, "324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf", id="64-default"
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

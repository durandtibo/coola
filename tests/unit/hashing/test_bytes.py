from __future__ import annotations

import pytest

from coola.hashing import BytesHasher, HasherRegistry, hash_bytes


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: BytesHasher()})


################################
#     Tests for BytesHasher    #
################################


def test_bytes_hasher_repr() -> None:
    assert repr(BytesHasher()) == "BytesHasher()"


def test_bytes_hasher_str() -> None:
    assert str(BytesHasher()) == "BytesHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            b"hello",
            "324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf",
            id="lowercase",
        ),
        pytest.param(
            b"Hello",
            "8b7ca7d27d9fc55fa30abfe515b3afb24e3fe89fdd02e2ac92bca2c96680642e",
            id="uppercase_first",
        ),
        pytest.param(
            b"hello ",
            "5c945bdc1b01abc46c3bee5681b8bd080157aa9e76a5a39c575d856aab41d5cf",
            id="trailing_space",
        ),
        pytest.param(
            b"",
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8",
            id="empty_bytes",
        ),
        pytest.param(
            b"Meowwwwww",
            "1b06bfa9e842b52eaf47386798687ccd22697ed0198cfda4e0eee7e4650595f5",
            id="meow",
        ),
    ],
)
def test_bytes_hasher_hash_parametrized(
    data: bytes, expected: str, registry: HasherRegistry
) -> None:
    assert BytesHasher().hash(data, registry=registry) == expected


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
def test_bytes_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = BytesHasher().hash(b"hello", registry=registry, length=length)
    assert result == expected
    assert len(result) == length


def test_bytes_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(BytesHasher().hash(b"hello", registry=registry), str)


def test_bytes_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hasher.hash(b"hello", registry=registry) == hasher.hash(b"hello", registry=registry)


def test_bytes_hasher_hash_is_case_sensitive(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hasher.hash(b"hello", registry=registry) != hasher.hash(b"Hello", registry=registry)


def test_bytes_hasher_hash_is_whitespace_sensitive(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hasher.hash(b"hello", registry=registry) != hasher.hash(b"hello ", registry=registry)


def test_bytes_hasher_hash_different_bytes_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = BytesHasher()
    assert hasher.hash(b"hello", registry=registry) != hasher.hash(b"world", registry=registry)


def test_bytes_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = BytesHasher()
    assert hasher.hash(b"hello", registry=registry) == hasher.hash(
        b"hello", registry=empty_registry
    )


###############################
#     Tests for hash_bytes    #
###############################

# --- Return type and format ---


def test_hash_bytes_returns_str() -> None:
    assert isinstance(hash_bytes(b"hello"), str)


def test_hash_bytes_returns_lowercase_hex() -> None:
    assert all(c in "0123456789abcdef" for c in hash_bytes(b"hello"))


def test_hash_bytes_default_length_is_64() -> None:
    assert len(hash_bytes(b"hello")) == 64


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
def test_hash_bytes_output_length_matches_requested(length: int) -> None:
    assert len(hash_bytes(b"hello", length=length)) == length


# --- Determinism ---


def test_hash_bytes_is_deterministic() -> None:
    assert hash_bytes(b"hello") == hash_bytes(b"hello")


def test_hash_bytes_empty_bytes_is_valid() -> None:
    result = hash_bytes(b"")
    assert isinstance(result, str)
    assert len(result) == 64


# --- Sensitivity ---


def test_hash_bytes_different_inputs_produce_different_hashes() -> None:
    assert hash_bytes(b"hello") != hash_bytes(b"world")


def test_hash_bytes_same_prefix_different_inputs_produce_different_hashes() -> None:
    assert hash_bytes(b"hello1") != hash_bytes(b"hello2")


def test_hash_bytes_is_case_sensitive() -> None:
    assert hash_bytes(b"hello") != hash_bytes(b"Hello")


def test_hash_bytes_is_sensitive_to_whitespace() -> None:
    assert hash_bytes(b"hello") != hash_bytes(b"hello ")


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
def test_hash_bytes_raises_for_invalid_length(length: int) -> None:
    with pytest.raises(ValueError, match=str(abs(length))):
        hash_bytes(b"hello", length=length)

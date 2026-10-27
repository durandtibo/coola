from __future__ import annotations

import pytest

from coola.hashing import BytesHasher, HasherRegistry, hash_bytes


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: BytesHasher()})


#################################
#     Tests for BytesHasher     #
#################################


def test_bytes_hasher_repr() -> None:
    assert repr(BytesHasher()) == "BytesHasher()"


def test_bytes_hasher_str() -> None:
    assert str(BytesHasher()) == "BytesHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            b"hello",
            "5445a0f97d2dd6ab7fe9478001593062297d6a661c6c991713726487c7a40d06",
            id="lowercase",
        ),
        pytest.param(
            b"Hello",
            "13e90b849db105c46ab09021e432a1fb86b2845628588382cb6027af8a437651",
            id="uppercase_first",
        ),
        pytest.param(
            b"hello ",
            "9eab563d356052f1aad497a2979000fee0926c58df699efced2f4a6968fd83b1",
            id="trailing_space",
        ),
        pytest.param(
            b"",
            "fd04376b68c6fb49d034171ad0b0d57f63c62df3347927f46f24bca0efb5abdc",
            id="empty_bytes",
        ),
        pytest.param(
            b"Meowwwwww",
            "7fed8b5db41d2fe23ee973445831a1de4862dbdbc3798fafeb13479e28e1ab12",
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
        pytest.param(16, "c03fc993641ae772", id="16"),
        pytest.param(32, "e95d1d684ca34b3cd6ddb88781f983e6", id="32"),
        pytest.param(
            64, "5445a0f97d2dd6ab7fe9478001593062297d6a661c6c991713726487c7a40d06", id="64-default"
        ),
    ],
)
def test_bytes_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = BytesHasher().hash(b"hello", registry=registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_bytes_hasher_hash_length_is_respected(length: int, registry: HasherRegistry) -> None:
    result = BytesHasher().hash(b"hello", registry=registry, length=length)
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


def test_bytes_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # BytesHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = BytesHasher()
    assert hasher.hash(b"hello", registry=registry) == hasher.hash(
        b"hello", registry=registry, ignore_unhashable=True
    )


def test_bytes_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = BytesHasher()
    assert hasher.hash(b"hello", registry=registry) == hasher.hash(
        b"hello", registry=empty_registry
    )


################################
#     Tests for hash_bytes     #
################################

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


def test_hash_bytes_empty_bytes_is_valid_length() -> None:
    result = hash_bytes(b"")
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


def test_bytes_hasher_hash_bytearray_differs_from_bytes(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hasher.hash(bytearray(b"abc"), registry=registry) != hasher.hash(
        b"abc", registry=registry
    )


def test_bytes_hasher_hash_bytearray_depends_on_content_same_content(
    registry: HasherRegistry,
) -> None:
    hasher = BytesHasher()
    assert hasher.hash(bytearray(b"abc"), registry=registry) == hasher.hash(
        bytearray(b"abc"), registry=registry
    )


def test_bytes_hasher_hash_bytearray_depends_on_content_different_content(
    registry: HasherRegistry,
) -> None:
    hasher = BytesHasher()
    assert hasher.hash(bytearray(b"abc"), registry=registry) != hasher.hash(
        bytearray(b"xyz"), registry=registry
    )


def test_bytes_hasher_hash_memoryview_differs_from_bytes(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hasher.hash(memoryview(b"abc"), registry=registry) != hasher.hash(
        b"abc", registry=registry
    )


def test_bytes_hasher_hash_memoryview_depends_on_content(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hasher.hash(memoryview(b"abc"), registry=registry) != hasher.hash(
        memoryview(b"xyz"), registry=registry
    )

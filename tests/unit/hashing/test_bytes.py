from __future__ import annotations

import pytest

from coola.hashing import BytesHasher, HasherRegistry, digest, tagged
from tests.unit.hashing.helpers import hash_with


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
            "c0849551a6d5dac1460074b366025b791f95050cabe8473b0b4f9770d4d138fa",
            id="lowercase",
        ),
        pytest.param(
            b"Hello",
            "292020a3ed29a425332b7bbb135895c6b5d45c44899e5cba853746c516463a5e",
            id="uppercase_first",
        ),
        pytest.param(
            b"hello ",
            "1ea576fab125b0927a421da5429b048386b50743b52c7a96001fe20dc35c1825",
            id="trailing_space",
        ),
        pytest.param(
            b"",
            "8c51179fe70c00c267a8efde9e8e22900a79d950843db47943c558d7fad41943",
            id="empty_bytes",
        ),
        pytest.param(
            b"Meowwwwww",
            "655c0b921474575a246ed9ed9fd003150bf74732c0158d47055d06705bf889fc",
            id="meow",
        ),
    ],
)
def test_bytes_hasher_hash_parametrized(
    data: bytes, expected: str, registry: HasherRegistry
) -> None:
    assert hash_with(BytesHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "c0849551a6d5dac1", id="16"),
        pytest.param(32, "c0849551a6d5dac1460074b366025b79", id="32"),
        pytest.param(
            64, "c0849551a6d5dac1460074b366025b791f95050cabe8473b0b4f9770d4d138fa", id="64-default"
        ),
    ],
)
def test_bytes_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(BytesHasher(), b"hello", registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_bytes_hasher_hash_length_is_respected(length: int, registry: HasherRegistry) -> None:
    result = hash_with(BytesHasher(), b"hello", registry, length=length)
    assert len(result) == length


def test_bytes_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(BytesHasher(), b"hello", registry), str)


def test_bytes_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, b"hello", registry) == hash_with(hasher, b"hello", registry)


def test_bytes_hasher_hash_is_case_sensitive(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, b"hello", registry) != hash_with(hasher, b"Hello", registry)


def test_bytes_hasher_hash_is_whitespace_sensitive(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, b"hello", registry) != hash_with(hasher, b"hello ", registry)


def test_bytes_hasher_hash_different_bytes_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, b"hello", registry) != hash_with(hasher, b"world", registry)


def test_bytes_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # BytesHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = BytesHasher()
    assert hash_with(hasher, b"hello", registry) == hash_with(
        hasher, b"hello", registry, ignore_unhashable=True
    )


def test_bytes_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = BytesHasher()
    assert hash_with(hasher, b"hello", registry) == hash_with(hasher, b"hello", empty_registry)


# --- Determinism ---


# --- Sensitivity ---


# --- Invalid length ---


def test_bytes_hasher_hash_bytearray_differs_from_bytes(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, bytearray(b"abc"), registry) != hash_with(hasher, b"abc", registry)


def test_bytes_hasher_hash_bytearray_depends_on_content_same_content(
    registry: HasherRegistry,
) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, bytearray(b"abc"), registry) == hash_with(
        hasher, bytearray(b"abc"), registry
    )


def test_bytes_hasher_hash_bytearray_depends_on_content_different_content(
    registry: HasherRegistry,
) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, bytearray(b"abc"), registry) != hash_with(
        hasher, bytearray(b"xyz"), registry
    )


def test_bytes_hasher_hash_memoryview_differs_from_bytes(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, memoryview(b"abc"), registry) != hash_with(hasher, b"abc", registry)


def test_bytes_hasher_hash_memoryview_depends_on_content(registry: HasherRegistry) -> None:
    hasher = BytesHasher()
    assert hash_with(hasher, memoryview(b"abc"), registry) != hash_with(
        hasher, memoryview(b"xyz"), registry
    )


def test_bytes_hasher_encode_is_tagged_with_the_digest_of_the_content() -> None:
    assert BytesHasher().encode(b"abc") == tagged("bytes", digest(b"abc"))


def test_bytes_hasher_encode_empty() -> None:
    assert BytesHasher().encode(b"") == tagged("bytes", digest(b""))


def test_bytes_hasher_encode_is_case_sensitive() -> None:
    assert BytesHasher().encode(b"a") != BytesHasher().encode(b"A")


def test_bytes_hasher_encode_is_sensitive_to_whitespace() -> None:
    assert BytesHasher().encode(b"a") != BytesHasher().encode(b"a ")

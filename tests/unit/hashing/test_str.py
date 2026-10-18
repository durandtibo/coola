from __future__ import annotations

from typing import Any

import pytest

from coola.hashing import HasherRegistry, ReprHasher, StrHasher, hash_tagged


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: StrHasher()})


###############################
#     Tests for StrHasher     #
###############################


def test_str_hasher_repr() -> None:
    assert repr(StrHasher()) == "StrHasher()"


def test_str_hasher_str() -> None:
    assert str(StrHasher()) == "StrHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            1234,
            "5361a39f90c3fbcbdbf28f3e75caaf46117ee724a45e2417293deb1c7372c3a5",
            id="int",
        ),
        pytest.param(
            0,
            "b48f4e205ce7c64358a63f4be6ca1b595a7938fa8c2957b93ae9226cdd3d25e0",
            id="int_zero",
        ),
        pytest.param(
            -1,
            "69f5d336ac4364e1572f7d644d39620039df1574057b1def70e8159d70a1443f",
            id="int_negative",
        ),
        pytest.param(
            1.5,
            "8449b0279dab0952bd63c08a8d6efbcc06acfbc8fcbdfb5bd994cbcb4592a76c",
            id="float",
        ),
        pytest.param(
            float("inf"),
            "60c2d38cf83713554ec16b91ff0f4368bed98a07ad6014bbcbeeae3b2b425111",
            id="float_inf",
        ),
        pytest.param(
            True,
            "f99a94bf0762f7954b0a911995be46edf4be9335c66e7b19ff46771a5721f224",
            id="bool_true",
        ),
        pytest.param(
            False,
            "35436762a2d50b51372325c767f9cd1135988b785179f539f7d8516955373616",
            id="bool_false",
        ),
        pytest.param(
            complex(1, 2),
            "ea65f84d990d47df43a7fb4a4c2448e226b3a4143801a78a977352c16bcdd24c",
            id="complex",
        ),
        pytest.param(
            complex(0, -1),
            "b03df2987589a8d12e8bde56c7901da39ecd5fcea8f4f82a40137b9ef91169e4",
            id="complex_imaginary",
        ),
        pytest.param(
            None,
            "16ce736f793e63bc86493579745ee0d01fddc8470fede131ab63c18fc703500e",
            id="none",
        ),
        pytest.param(
            "hello",
            "7258221889359377b94d9af3fddce64bb5a41be7dd2547bf259cedc7bc0c2ceb",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "27f6ffcf7a27a7c5e4a693584aec688b7b2ad5ecac4e4c58aec67bc1f3030aa4",
            id="list",
        ),
        pytest.param(
            {"a": 1},
            "a96506e297db564d1a88d3f06d73ec708a62c598b2bceba7fc00e5067a6cd29b",
            id="dict",
        ),
        pytest.param(
            (1, 2, 3),
            "3d9feb22294eb0a8b1278e2e412e29dd8cc17dd8431113574e4f1572fd64ed06",
            id="tuple",
        ),
        pytest.param(
            b"bytes",
            "403df08c312ea6bec985ab16f1c7f941976991e4766a9f0a7a24077e9a0a5b9a",
            id="bytes",
        ),
    ],
)
def test_str_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert StrHasher().hash(data, registry=registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "9fc476b447085da8", id="16"),
        pytest.param(32, "de01d2683340090c5780e0ff4a4035d4", id="32"),
        pytest.param(
            64, "7258221889359377b94d9af3fddce64bb5a41be7dd2547bf259cedc7bc0c2ceb", id="64-default"
        ),
    ],
)
def test_str_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = StrHasher().hash("hello", registry=registry, length=length)
    assert result == expected
    assert len(result) == length


def test_str_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(StrHasher().hash("hello", registry=registry), str)


def test_str_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = StrHasher()
    assert hasher.hash("hello", registry=registry) == hasher.hash("hello", registry=registry)


def test_str_hasher_hash_different_values_different_hashes(registry: HasherRegistry) -> None:
    hasher = StrHasher()
    assert hasher.hash(1, registry=registry) != hasher.hash(2, registry=registry)


def test_str_hasher_hash_bool_and_int_different_hashes(registry: HasherRegistry) -> None:
    # str(True) == 'True' and str(1) == '1', so they must not collide.
    hasher = StrHasher()
    assert hasher.hash(True, registry=registry) != hasher.hash(1, registry=registry)


def test_str_hasher_hash_uses_str_not_repr(registry: HasherRegistry) -> None:
    # For objects where str() and repr() differ, StrHasher must use str().
    # str("hello") == 'hello' while repr("hello") == "'hello'".
    class MyObj:
        def __str__(self) -> str:
            return "str_val"

        def __repr__(self) -> str:
            return "repr_val"

    obj = MyObj()
    hasher = StrHasher()
    tag = f"str:{MyObj.__qualname__}"
    assert hasher.hash(obj, registry=registry) == hash_tagged(tag, str(obj))
    assert hasher.hash(obj, registry=registry) != hash_tagged(tag, repr(obj))


def test_str_hasher_hash_str_differs_from_repr_hasher_for_strings(
    registry: HasherRegistry,
) -> None:
    # StrHasher hashes str("hello") = "hello".
    # ReprHasher hashes repr("hello") = "'hello'" (with quotes).
    # The two must produce different results for string inputs.

    hasher_str = StrHasher()
    hasher_repr = ReprHasher()
    assert hasher_str.hash("hello", registry=registry) != hasher_repr.hash(
        "hello", registry=registry
    )


def test_str_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # StrHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = StrHasher()
    assert hasher.hash("hello", registry=registry) == hasher.hash(
        "hello", registry=registry, ignore_unhashable=True
    )


def test_str_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = StrHasher()
    assert hasher.hash("hello", registry=registry) == hasher.hash("hello", registry=empty_registry)

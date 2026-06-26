from __future__ import annotations

from typing import Any

import pytest

from coola.hashing import HasherRegistry, ReprHasher, StrHasher, hash_string


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
            "bf1003cd5c1336387f7e4eebf72a3d9cd4fa8ab5be19825bc0e3ecd8ce1cd140",
            id="int",
        ),
        pytest.param(
            0,
            "0fd923ca5e7218c4ba3c3801c26a617ecdbfdaebb9c76ce2eca166e7855efbb8",
            id="int_zero",
        ),
        pytest.param(
            -1,
            "63ac695b1ed4698e9ab1e2aee9b9802b584e65810c58a3656440411d713b8ce2",
            id="int_negative",
        ),
        pytest.param(
            1.5,
            "92a0f8f152d1206df5b494cdbcec4de630269bda0df7f59a9a96a4fc5cc6e955",
            id="float",
        ),
        pytest.param(
            float("inf"),
            "094ec93925e633ab5b33fa4dba41328559b931c8f59d255753c3e14510c0f6d8",
            id="float_inf",
        ),
        pytest.param(
            True,
            "b37c53228410790b2e6d4ab6eb00deb4e1e9b47e2075100b120e0abd777d0020",
            id="bool_true",
        ),
        pytest.param(
            False,
            "caf16c1ae983a5041d15f62141676b96db1f7f850ea7ef2f62f1a01a469b9c7c",
            id="bool_false",
        ),
        pytest.param(
            complex(1, 2),
            "672e535e27fd6becac3687e261cc67a9708673e67eac19d588d574efadafc1e0",
            id="complex",
        ),
        pytest.param(
            complex(0, -1),
            "ac607ed8de2919e6705d9dbda2ada4605384cb27e0c03e1c1d272d49dec6f1ab",
            id="complex_imaginary",
        ),
        pytest.param(
            None,
            "047380c75d8da0e84df15f8218632f31ca7058142d75fa4ee225aea3e8b1da82",
            id="none",
        ),
        pytest.param(
            "hello",
            "324dcf027dd4a30a932c441f365a25e86b173defa4b8e58948253471b81b72cf",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "07d1c573fdfe5074241be58b56a31220ff7a9cb3dd94f1cba650eca27c0a421e",
            id="list",
        ),
        pytest.param(
            {"a": 1},
            "1fda8e5b5936528de9a77d335493f7299d258f78417cf6e69f737a439129cb68",
            id="dict",
        ),
        pytest.param(
            (1, 2, 3),
            "a0906786d40534291e1543b93de2faae35d9328bea5182f925e276fcedfb81ae",
            id="tuple",
        ),
        pytest.param(
            b"bytes",
            "e0652d4bc0821a8be21ad001b60a4799617edf4fd26c5396152784301f86c59f",
            id="bytes",
        ),
    ],
)
def test_str_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert StrHasher().hash(data, registry=registry) == expected


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
    assert hasher.hash(obj, registry=registry) == hash_string(str(obj))
    assert hasher.hash(obj, registry=registry) != hash_string(repr(obj))


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


def test_str_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = StrHasher()
    assert hasher.hash("hello", registry=registry) == hasher.hash("hello", registry=empty_registry)

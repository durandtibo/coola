from __future__ import annotations

from typing import Any

import pytest

from coola.hashing import HasherRegistry, ReprHasher, StringHasher


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry({object: ReprHasher()})


################################
#     Tests for ReprHasher     #
################################


def test_repr_hasher_repr() -> None:
    assert repr(ReprHasher()) == "ReprHasher()"


def test_repr_hasher_str() -> None:
    assert str(ReprHasher()) == "ReprHasher()"


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
    ],
)
def test_repr_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert ReprHasher().hash(data, registry=registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "73b85de28f512fd3", id="16"),
        pytest.param(32, "a231498f6c1f441aa98482ea0b224ffa", id="32"),
        pytest.param(
            64, "bf1003cd5c1336387f7e4eebf72a3d9cd4fa8ab5be19825bc0e3ecd8ce1cd140", id="64-default"
        ),
    ],
)
def test_repr_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = ReprHasher().hash(1234, registry=registry, length=length)
    assert result == expected
    assert len(result) == length


def test_repr_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(ReprHasher().hash(1234, registry=registry), str)


def test_repr_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hasher.hash(1234, registry=registry) == hasher.hash(1234, registry=registry)


def test_repr_hasher_hash_different_values_different_hashes(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hasher.hash(1, registry=registry) != hasher.hash(2, registry=registry)


def test_repr_hasher_hash_bool_and_int_different_hashes(registry: HasherRegistry) -> None:
    # repr(True) == 'True' and repr(1) == '1', so they must not collide.
    hasher = ReprHasher()
    assert hasher.hash(True, registry=registry) != hasher.hash(1, registry=registry)


def test_repr_hasher_hash_uses_repr_not_str(registry: HasherRegistry) -> None:
    # For objects where repr() and str() differ, ReprHasher must use repr().
    class MyObj:
        def __str__(self) -> str:
            return "str_val"

        def __repr__(self) -> str:
            return "repr_val"

    obj = MyObj()
    hasher = ReprHasher()
    assert hasher.hash(obj, registry=registry) != StringHasher().hash(str(obj), registry=registry)
    assert hasher.hash(obj, registry=registry) == StringHasher().hash(repr(obj), registry=registry)


def test_repr_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = ReprHasher()
    assert hasher.hash(1234, registry=registry) == hasher.hash(1234, registry=empty_registry)

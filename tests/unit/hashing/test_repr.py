from __future__ import annotations

from typing import Any

import pytest

from coola.hashing import HasherRegistry, ReprHasher, hash_tagged
from coola.testing.fixtures import numpy_available


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
            "813efc59055b35ec46d4e6971b8c90e67b18f8b0cacd953e0d70493327f568a5",
            id="int",
        ),
        pytest.param(
            0,
            "07f0eb8b3e8eb4fee6a02c0dc4c857d883793ed1d4fdff4e4cce1ed0bb9bc90c",
            id="int_zero",
        ),
        pytest.param(
            -1,
            "178ab2776b8954c1cf2805104d2582a0f974ad5dee25a0bd0d12f0791969e70c",
            id="int_negative",
        ),
        pytest.param(
            1.5,
            "d7f06307c8bc1fc0007627151ef4218c1380c4f83507367e5ffcaa908fae3f40",
            id="float",
        ),
        pytest.param(
            float("inf"),
            "93b281324bf786c6cef88cdd98f0315df306f8d5b5e62749a3377efcfbfd0944",
            id="float_inf",
        ),
        pytest.param(
            True,
            "d7d4d5d42aa13a2cb19b77506ff0e2cfe278713fecddd620b898d65bc7dc63dd",
            id="bool_true",
        ),
        pytest.param(
            False,
            "901ccd048a2483766d509296bdbb1d9863cd49e807968da74103fc1b33d52921",
            id="bool_false",
        ),
        pytest.param(
            complex(1, 2),
            "4ee227f701d6a0eaa2429bfb3378dd865abe0e5fc9d303068c0befaffbab41b1",
            id="complex",
        ),
        pytest.param(
            complex(0, -1),
            "34d318d9f41ef50c8a2b4985683cbd6bf7efdbe279d18c70efcccef62730ea04",
            id="complex_imaginary",
        ),
        pytest.param(
            None,
            "633466e4907837d7b866f9a81e2569b137155569e183d55f94b969e4becf2ebe",
            id="none",
        ),
        pytest.param(
            "hello",
            "99c8f33e3377a9422a4845c80a7f46d896ae5f08954b2a462420af63c0871549",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "127538d692be01becf3979a9471bae50668ad68398df1197c246d5de213c3576",
            id="list",
        ),
        pytest.param(
            {"a": 1},
            "8b779e67c126a8c758f22c73c9365bfbe1ba81881d4db992b9c481a05a62b113",
            id="dict",
        ),
        pytest.param(
            (1, 2, 3),
            "7ea790fd3c5ebe3e9fd915fb0a860f85665e6773f000aa9f11d7c82f02e606c1",
            id="tuple",
        ),
        pytest.param(
            b"bytes",
            "d6609413d3c33ebac4d08b8046c811a7e9d6ed552b7c8c557cb53b53ec0b85d5",
            id="bytes",
        ),
    ],
)
def test_repr_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert ReprHasher().hash(data, registry=registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "eb848ef1723310c1", id="16"),
        pytest.param(32, "7cc3d9ac0d2e7aed62365f475e476cd1", id="32"),
        pytest.param(
            64, "813efc59055b35ec46d4e6971b8c90e67b18f8b0cacd953e0d70493327f568a5", id="64-default"
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
    tag = f"{type(obj).__module__}.{type(obj).__qualname__}"
    assert hasher.hash(obj, registry=registry) != hash_tagged(tag, str(obj))
    assert hasher.hash(obj, registry=registry) == hash_tagged(tag, repr(obj))


def test_repr_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # ReprHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = ReprHasher()
    assert hasher.hash(1234, registry=registry) == hasher.hash(
        1234, registry=registry, ignore_unhashable=True
    )


def test_repr_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = ReprHasher()
    assert hasher.hash(1234, registry=registry) == hasher.hash(1234, registry=empty_registry)


def test_repr_hasher_hash_negative_zero_equals_zero(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hasher.hash(-0.0, registry=registry) == hasher.hash(0.0, registry=registry)


def test_repr_hasher_hash_negative_zero_complex_components(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hasher.hash(complex(-0.0, -0.0), registry=registry) == hasher.hash(0j, registry=registry)
    assert hasher.hash(complex(-0.0, 1.0), registry=registry) == hasher.hash(
        complex(0.0, 1.0), registry=registry
    )


def test_repr_hasher_hash_zero_differs_from_one(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hasher.hash(0.0, registry=registry) != hasher.hash(1.0, registry=registry)


@numpy_available
def test_repr_hasher_numpy_scalar_without_builtin_equivalent(registry: HasherRegistry) -> None:
    import numpy as np

    value = np.datetime64("2020-01-01")
    assert ReprHasher().hash(value, registry) == hash_tagged("numpy.datetime64", repr(value))


def test_repr_hasher_hash_user_class_does_not_share_the_tag_of_a_builtin_name(
    registry: HasherRegistry,
) -> None:
    class Decimal:
        def __repr__(self) -> str:
            return "Decimal('1')"

    hasher = ReprHasher()
    assert hasher.hash(Decimal(), registry) != hash_tagged("Decimal", "Decimal('1')")


def test_repr_hasher_hash_builtin_class_keeps_the_bare_name(registry: HasherRegistry) -> None:
    assert ReprHasher().hash(None, registry) == hash_tagged("NoneType", "None")

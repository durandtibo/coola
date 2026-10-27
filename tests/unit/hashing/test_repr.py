from __future__ import annotations

import sys
from datetime import timedelta
from enum import Enum
from typing import Any
from unittest.mock import patch

import pytest

from coola.hashing import HasherRegistry, ReprHasher, tagged
from coola.hashing.repr import _numpy_scalar_to_builtin
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available
from tests.unit.hashing.helpers import hash_with, leaf_hash

if is_numpy_available():
    import numpy as np


class MixedInt(int, Enum):
    ONE = 1


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
            "f6c0da974e8e498878127df2176ce818d5b1dd1f109d364635f0ee4ff5d0b76f",
            id="int",
        ),
        pytest.param(
            0,
            "55ca561f078b8ee39c12bf706a975da9bd93714cd8d9d4e7c4fbfab330ff0bd2",
            id="int_zero",
        ),
        pytest.param(
            -1,
            "de3827b566bd4de0e65c9ae59680f8f235cd324dbf9e5a3c7db4663d31414685",
            id="int_negative",
        ),
        pytest.param(
            1.5,
            "5ae23551bd86d7cd566f2c064292e3511a0c4077417627d6ec1f45d63b88f960",
            id="float",
        ),
        pytest.param(
            float("inf"),
            "77e5af088c11e9a6f7998ba7e3c7482f17fd2c4908a81be6a851c89c669dd0b9",
            id="float_inf",
        ),
        pytest.param(
            True,
            "b125316ff709ad952976c214852acc80ef2e75dd12dd50a908a204705d373820",
            id="bool_true",
        ),
        pytest.param(
            False,
            "9958f9ec91be1d950bf8bbce8d9210df87445fa139b8afe6f8aa4900579a4ea6",
            id="bool_false",
        ),
        pytest.param(
            complex(1, 2),
            "475b70f206b62fc93989cfc073a63c05b23accc90b082d77d984f4c3c87b1862",
            id="complex",
        ),
        pytest.param(
            complex(0, -1),
            "5b354564a5c980427ce1193beb6905a6d6778c25cfaa4e69c0c7bad0873d9625",
            id="complex_imaginary",
        ),
        pytest.param(
            None,
            "f8a2900c635ef9c70d6069a2c035ee698c3f90761c9b302e412a7a36da951ada",
            id="none",
        ),
        pytest.param(
            "hello",
            "a8342aed984ebfd37f29c5c93f45f9e78b833a126929452b394dae3d4ffcfc73",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "7ce70d150ad3b1f5bfa45e5fb3aac39c9f03a720916c2e8478c085a225828012",
            id="list",
        ),
        pytest.param(
            {"a": 1},
            "fa58be2a8064c94238cbc3e995631feecc952cf3ed58a9b093be172432da77c0",
            id="dict",
        ),
        pytest.param(
            (1, 2, 3),
            "0204d5e38046b657b6e5fc9ff2e2364447a4cc84260d5cef0f214a33d24af5c5",
            id="tuple",
        ),
        pytest.param(
            b"bytes",
            "bfbd7c6379a32f43baed60a8caa5d82a02f30e0a41bfeec239552fc177a4e699",
            id="bytes",
        ),
    ],
)
def test_repr_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert hash_with(ReprHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "f6c0da974e8e4988", id="16"),
        pytest.param(32, "f6c0da974e8e498878127df2176ce818", id="32"),
        pytest.param(
            64, "f6c0da974e8e498878127df2176ce818d5b1dd1f109d364635f0ee4ff5d0b76f", id="64-default"
        ),
    ],
)
def test_repr_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(ReprHasher(), 1234, registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_repr_hasher_hash_length_length_is_respected(length: int, registry: HasherRegistry) -> None:
    result = hash_with(ReprHasher(), 1234, registry, length=length)
    assert len(result) == length


def test_repr_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(ReprHasher(), 1234, registry), str)


def test_repr_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, 1234, registry) == hash_with(hasher, 1234, registry)


def test_repr_hasher_hash_different_values_different_hashes(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, 1, registry) != hash_with(hasher, 2, registry)


def test_repr_hasher_hash_bool_and_int_different_hashes(registry: HasherRegistry) -> None:
    # repr(True) == 'True' and repr(1) == '1', so they must not collide.
    hasher = ReprHasher()
    assert hash_with(hasher, True, registry) != hash_with(hasher, 1, registry)


def test_repr_hasher_hash_uses_repr_not_str_equals_the_repr_hash(registry: HasherRegistry) -> None:
    # For objects where repr() and str() differ, ReprHasher must use repr().
    class MyObj:
        def __str__(self) -> str:
            return "str_val"

        def __repr__(self) -> str:
            return "repr_val"

    obj = MyObj()
    hasher = ReprHasher()
    tag = f"{type(obj).__module__}.{type(obj).__qualname__}"
    assert hash_with(hasher, obj, registry) == leaf_hash(tagged(tag, repr(obj)))


def test_repr_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # ReprHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = ReprHasher()
    assert hash_with(hasher, 1234, registry) == hash_with(
        hasher, 1234, registry, ignore_unhashable=True
    )


def test_repr_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = ReprHasher()
    assert hash_with(hasher, 1234, registry) == hash_with(hasher, 1234, empty_registry)


def test_repr_hasher_hash_negative_zero_equals_zero(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, -0.0, registry) == hash_with(hasher, 0.0, registry)


def test_repr_hasher_hash_negative_zero_complex_components_zero(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, complex(-0.0, -0.0), registry) == hash_with(hasher, 0j, registry)


def test_repr_hasher_hash_negative_zero_complex_components_imaginary_part(
    registry: HasherRegistry,
) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, complex(-0.0, 1.0), registry) == hash_with(
        hasher, complex(0.0, 1.0), registry
    )


def test_repr_hasher_hash_zero_differs_from_one(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, 0.0, registry) != hash_with(hasher, 1.0, registry)


@numpy_available
def test_repr_hasher_numpy_scalar_without_builtin_equivalent(registry: HasherRegistry) -> None:
    value = np.datetime64("2020-01-01")
    assert hash_with(ReprHasher(), value, registry) == leaf_hash(
        tagged("numpy.datetime64", repr(value))
    )


def test_repr_hasher_hash_builtin_class_keeps_the_bare_name(registry: HasherRegistry) -> None:
    assert hash_with(ReprHasher(), None, registry) == leaf_hash(tagged("NoneType", "None"))


def test_repr_hasher_hash_int_above_str_conversion_limit(registry: HasherRegistry) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, 10**5000, registry) == hash_with(hasher, 10**5000, registry)


def test_repr_hasher_hash_int_above_str_conversion_limit_different_values(
    registry: HasherRegistry,
) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, 10**5000, registry) != hash_with(hasher, 10**5000 + 1, registry)


def test_repr_hasher_hash_int_above_str_conversion_limit_negative(
    registry: HasherRegistry,
) -> None:
    hasher = ReprHasher()
    assert hash_with(hasher, -(10**5000), registry) != hash_with(hasher, 10**5000, registry)


##########################################
#     Tests for _numpy_scalar_to_builtin #
##########################################


def test_numpy_scalar_to_builtin_does_not_import_numpy() -> None:
    value = object()
    with patch.dict(sys.modules, {"numpy": None}):
        assert _numpy_scalar_to_builtin(value) is value


def test_repr_hasher_repr_with_tag() -> None:
    assert repr(ReprHasher("timedelta")) == "ReprHasher(tag='timedelta')"


def test_repr_hasher_hash_explicit_tag(registry: HasherRegistry) -> None:
    assert hash_with(ReprHasher("timedelta"), timedelta(days=1), registry) == leaf_hash(
        tagged("timedelta", repr(timedelta(days=1)))
    )


def test_repr_hasher_hash_explicit_tag_keeps_the_builtin_scalars(registry: HasherRegistry) -> None:
    assert hash_with(ReprHasher("timedelta"), 1, registry) == hash_with(ReprHasher(), 1, registry)


def test_repr_hasher_encode_user_class_does_not_share_the_tag_of_a_builtin_name() -> None:
    class int:  # noqa: A001, N801
        def __repr__(self) -> str:
            return "1"

    assert ReprHasher().encode(int()) != ReprHasher().encode(1)

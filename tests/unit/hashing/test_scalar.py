from __future__ import annotations

import sys
from collections import UserString
from datetime import timedelta
from enum import Enum, Flag, IntEnum, StrEnum
from typing import Any
from unittest.mock import patch

import pytest

from coola.hashing import (
    BytesHasher,
    EnumHasher,
    HasherRegistry,
    ReprHasher,
    StrHasher,
    StringHasher,
    digest,
    get_default_registry,
    tagged,
)
from coola.hashing.scalar import _numpy_scalar_to_builtin
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available
from tests.unit.hashing.helpers import hash_with, leaf_hash


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry()


class MixedStr(str, Enum):  # noqa: UP042
    A = "a"


##################################
#     Tests for StringHasher     #
##################################


def test_string_hasher_repr() -> None:
    assert repr(StringHasher()) == "StringHasher()"


def test_string_hasher_str() -> None:
    assert str(StringHasher()) == "StringHasher()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            "hello",
            "2998e64e4b20ac91774f304f5527a5160d30249cdcdc984effac931cf54196e0",
            id="lowercase",
        ),
        pytest.param(
            "Hello",
            "d2e6b55494e319a95d4724ef225b9c8fa1a8cecae70f33f9b4421bfc1703dbee",
            id="uppercase_first",
        ),
        pytest.param(
            "hello ",
            "58629ea5bbc011db083e224e27e7dcc4b88c747964447d3fb51ab344fcc42a89",
            id="trailing_space",
        ),
        pytest.param(
            "",
            "13794f6a993bed50b22118af38aaa84e510b29e67ee82abb91bc758b9b8ba4d3",
            id="empty_string",
        ),
        pytest.param(
            "Meowwwwww",
            "dc79d6439fb717960d695a964b67a3c46f68e692b07cc1b9e59828d6a9299f10",
            id="meow",
        ),
    ],
)
def test_string_hasher_hash_parametrized(
    data: str, expected: str, registry: HasherRegistry
) -> None:
    assert hash_with(StringHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "2998e64e4b20ac91", id="16"),
        pytest.param(32, "2998e64e4b20ac91774f304f5527a516", id="32"),
        pytest.param(
            64, "2998e64e4b20ac91774f304f5527a5160d30249cdcdc984effac931cf54196e0", id="64-default"
        ),
    ],
)
def test_string_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(StringHasher(), "hello", registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_string_hasher_hash_length_length_is_respected(
    length: int, registry: HasherRegistry
) -> None:
    result = hash_with(StringHasher(), "hello", registry, length=length)
    assert len(result) == length


def test_string_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(StringHasher(), "hello", registry), str)


def test_string_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(hasher, "hello", registry)


def test_string_hasher_hash_is_case_sensitive(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) != hash_with(hasher, "Hello", registry)


def test_string_hasher_hash_is_whitespace_sensitive(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) != hash_with(hasher, "hello ", registry)


def test_string_hasher_hash_different_strings_different_hashes(
    registry: HasherRegistry,
) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) != hash_with(hasher, "world", registry)


def test_string_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # StringHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(
        hasher, "hello", registry, ignore_unhashable=True
    )


def test_string_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = StringHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(hasher, "hello", empty_registry)


# --- Determinism ---


# --- Sensitivity ---


# --- Invalid length ---


def test_string_hasher_hash_is_tagged() -> None:
    assert hash_with(StringHasher(), "hello", HasherRegistry()) == leaf_hash(tagged("str", "hello"))


##########################################
#     Tests for validate_hash_length     #
##########################################


def test_string_hasher_hash_user_string_differs_from_str(registry: HasherRegistry) -> None:
    hasher = StringHasher()
    assert hash_with(hasher, UserString("abc"), registry) != hash_with(hasher, "abc", registry)


def test_string_hasher_hash_user_string_equals_the_tagged_text(registry: HasherRegistry) -> None:
    assert hash_with(StringHasher(), UserString("abc"), registry) == leaf_hash(
        tagged("UserString", "abc")
    )


def test_string_hasher_encode_valid_utf8() -> None:
    assert StringHasher().encode("é") == b"str\x00\xc3\xa9"


def test_string_hasher_encode_lone_surrogate_differs_from_the_replacement_character() -> None:
    assert StringHasher().encode("\ud800") != StringHasher().encode("\ufffd")


def test_string_hasher_encode_lone_surrogates_differ() -> None:
    assert StringHasher().encode("\ud800") != StringHasher().encode("\ud801")


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


class Color(Enum):
    RED = 1
    BLUE = 2


class Shade(Enum):
    RED = 1


class Number(IntEnum):
    ONE = 1


class Label(StrEnum):
    A = "a"


class Perm(Flag):
    R = 1
    W = 2


class MixedInt(int, Enum):
    ONE = 1


################################
#     Tests for EnumHasher     #
################################


def test_enum_hasher_repr() -> None:
    assert repr(EnumHasher()) == "EnumHasher()"


def test_enum_hasher_str() -> None:
    assert str(EnumHasher()) == "EnumHasher()"


def test_enum_hasher_hash_is_the_hash_of_the_leaf(registry: HasherRegistry) -> None:
    assert hash_with(EnumHasher(), Color.RED, registry) == leaf_hash(EnumHasher().encode(Color.RED))


@pytest.mark.parametrize("length", [2, 16, 64, 128])
def test_enum_hasher_hash_length(length: int, registry: HasherRegistry) -> None:
    assert len(hash_with(EnumHasher(), Color.RED, registry, length=length)) == length


def test_enum_hasher_hash_invalid_length_raises(registry: HasherRegistry) -> None:
    with pytest.raises(ValueError, match="length must be an even number"):
        hash_with(EnumHasher(), Color.RED, registry, length=3)


###################################################
#     Tests for enums in the default registry     #
###################################################


@pytest.mark.parametrize("member", [Color.RED, Number.ONE, Label.A, MixedStr.A, MixedInt.ONE])
def test_default_registry_hashes_enum_members_as_enum(member: Enum) -> None:
    assert get_default_registry().hash(member) == leaf_hash(EnumHasher().encode(member))


def test_default_registry_enum_is_not_collapsed_into_value_int_enum() -> None:
    registry = get_default_registry()
    assert registry.hash(Number.ONE) != registry.hash(1)


def test_default_registry_enum_is_not_collapsed_into_value_str_enum() -> None:
    registry = get_default_registry()
    assert registry.hash(Label.A) != registry.hash("a")


def test_default_registry_enum_is_not_collapsed_into_value_mixed_str_enum() -> None:
    registry = get_default_registry()
    assert registry.hash(MixedStr.A) != registry.hash("a")


def test_default_registry_enum_is_not_collapsed_into_value_mixed_int_enum() -> None:
    registry = get_default_registry()
    assert registry.hash(MixedInt.ONE) != registry.hash(1)


class MixedBytes(bytes, Enum):
    A = b"a"


class MixedTuple(tuple, Enum):
    A = (1, 2)


@pytest.mark.parametrize("member", [MixedBytes.A, MixedTuple.A])
def test_default_registry_hashes_enum_mixins_as_enum(member: Enum) -> None:
    assert get_default_registry().hash(member) == leaf_hash(EnumHasher().encode(member))


@pytest.mark.parametrize("member", [MixedBytes.A, MixedTuple.A])
def test_default_registry_hashes_enum_mixins_differently_from_the_value(member: Enum) -> None:
    registry = get_default_registry()
    assert registry.hash(member) != registry.hash(member.value)


def test_enum_hasher_encode_different_members_differ() -> None:
    assert EnumHasher().encode(Color.RED) != EnumHasher().encode(Color.BLUE)


def test_enum_hasher_encode_same_member_name_in_different_classes_differ() -> None:
    assert EnumHasher().encode(Color.RED) != EnumHasher().encode(Shade.RED)


def test_enum_hasher_encode_composite_flag_differs_from_its_members() -> None:
    hasher = EnumHasher()
    assert hasher.encode(Perm.R | Perm.W) not in {hasher.encode(Perm.R), hasher.encode(Perm.W)}


def test_enum_hasher_encode_is_the_qualified_member_name() -> None:
    assert EnumHasher().encode(Color.RED) == f"Enum\x00{__name__}.Color.RED".encode()


if is_numpy_available():
    import numpy as np


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
            "44cb4c7822fa217c5c06c223c5b0aacc8c5c77c768df58591c091cb1500d815e",
            id="int",
        ),
        pytest.param(
            0,
            "aa89b5cee92948ed5b79726bbe6ee622716714cb89d6a929b74e576c8f0ca378",
            id="int_zero",
        ),
        pytest.param(
            -1,
            "ed8e02ad8b7c52faf9cfc853e54b3458a897bac0fa4cba0adfaa7b3cfe625851",
            id="int_negative",
        ),
        pytest.param(
            1.5,
            "d2c7ed87af69f0e55c61e9c0ac0f7f542447e494741cb8abc05168c9af2915c0",
            id="float",
        ),
        pytest.param(
            float("inf"),
            "237ebc3ac1cb090ca1513f9c741b1dcefae94a966e596e108ad9ebf562e98fef",
            id="float_inf",
        ),
        pytest.param(
            True,
            "a1246624c11215e07f0539a1d087c88b9a5f9c80a51424fac2ae1653cb91c9d6",
            id="bool_true",
        ),
        pytest.param(
            False,
            "6070e87c818c9d5527cc5b6d746bc073c252fe0f0ba6289e0e412e855005cc7a",
            id="bool_false",
        ),
        pytest.param(
            complex(1, 2),
            "e8f0895d7008e0922301b7a8802593794f7ab11f685fb7341fd705ebad5f3375",
            id="complex",
        ),
        pytest.param(
            complex(0, -1),
            "a52d76525628375f316e05c2b99b63f6448b087bb37981bb13ad815f3c271604",
            id="complex_imaginary",
        ),
        pytest.param(
            None,
            "a0fa4225f8e9bd67116bcc6996d3f48b888caade696d53d26cbdf44179e8ea61",
            id="none",
        ),
        pytest.param(
            "hello",
            "c548d0f1ffb93319621f7004d9155e45eebfb2726ef58d4bf899944c6d1c0dc2",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "8692ee439fcd517d39ce2e404a01f2fcc108a78ab7097301bb01ed452c96348d",
            id="list",
        ),
        pytest.param(
            {"a": 1},
            "23f399661da961d8049800fdfcdead0e209e54ac9987f1a8f18eacd24ddc6064",
            id="dict",
        ),
        pytest.param(
            (1, 2, 3),
            "6bb67c32d839f45ca6c2a9420c0f5550225505eb9cabec25b67fd16ebd0f7df7",
            id="tuple",
        ),
        pytest.param(
            b"bytes",
            "1ab8454f86ef06979217c00d3887954f7b9488c8c5485e803b03398d8a35f6a7",
            id="bytes",
        ),
    ],
)
def test_str_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert hash_with(StrHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "c548d0f1ffb93319", id="16"),
        pytest.param(32, "c548d0f1ffb93319621f7004d9155e45", id="32"),
        pytest.param(
            64, "c548d0f1ffb93319621f7004d9155e45eebfb2726ef58d4bf899944c6d1c0dc2", id="64-default"
        ),
    ],
)
def test_str_hasher_hash_length(length: int, expected: str, registry: HasherRegistry) -> None:
    result = hash_with(StrHasher(), "hello", registry, length=length)
    assert result == expected


@pytest.mark.parametrize("length", [16, 32, 64])
def test_str_hasher_hash_length_length_is_respected(length: int, registry: HasherRegistry) -> None:
    result = hash_with(StrHasher(), "hello", registry, length=length)
    assert len(result) == length


def test_str_hasher_hash_returns_str(registry: HasherRegistry) -> None:
    assert isinstance(hash_with(StrHasher(), "hello", registry), str)


def test_str_hasher_hash_is_deterministic(registry: HasherRegistry) -> None:
    hasher = StrHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(hasher, "hello", registry)


def test_str_hasher_hash_different_values_different_hashes(registry: HasherRegistry) -> None:
    hasher = StrHasher()
    assert hash_with(hasher, 1, registry) != hash_with(hasher, 2, registry)


def test_str_hasher_hash_bool_and_int_different_hashes(registry: HasherRegistry) -> None:
    # str(True) == 'True' and str(1) == '1', so they must not collide.
    hasher = StrHasher()
    assert hash_with(hasher, True, registry) != hash_with(hasher, 1, registry)


def test_str_hasher_hash_uses_str_not_repr_equals_the_str_hash(registry: HasherRegistry) -> None:
    # For objects where str() and repr() differ, StrHasher must use str().
    # str("hello") == 'hello' while repr("hello") == "'hello'".
    class MyObj:
        def __str__(self) -> str:
            return "str_val"

        def __repr__(self) -> str:
            return "repr_val"

    obj = MyObj()
    hasher = StrHasher()
    tag = f"str:{MyObj.__module__}.{MyObj.__qualname__}"
    assert hash_with(hasher, obj, registry) == leaf_hash(tagged(tag, str(obj)))


def test_str_hasher_hash_str_differs_from_repr_hasher_for_strings(
    registry: HasherRegistry,
) -> None:
    # StrHasher hashes str("hello") = "hello".
    # ReprHasher hashes repr("hello") = "'hello'" (with quotes).
    # The two must produce different results for string inputs.

    hasher_str = StrHasher()
    hasher_repr = ReprHasher()
    assert hash_with(hasher_str, "hello", registry) != hash_with(hasher_repr, "hello", registry)


def test_str_hasher_hash_ignore_unhashable_has_no_effect(registry: HasherRegistry) -> None:
    # StrHasher is a leaf hasher — ignore_unhashable is accepted for
    # interface consistency but has no effect since it never recurses.
    hasher = StrHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(
        hasher, "hello", registry, ignore_unhashable=True
    )


def test_str_hasher_hash_does_not_use_registry(registry: HasherRegistry) -> None:
    empty_registry = HasherRegistry()
    hasher = StrHasher()
    assert hash_with(hasher, "hello", registry) == hash_with(hasher, "hello", empty_registry)


def test_str_hasher_repr_with_tag() -> None:
    assert repr(StrHasher("UUID")) == "StrHasher(tag='UUID')"


def test_str_hasher_hash_explicit_tag(registry: HasherRegistry) -> None:
    assert hash_with(StrHasher("UUID"), 1234, registry) == leaf_hash(tagged("UUID", "1234"))


def test_str_hasher_hash_explicit_tag_applies_to_subclasses(registry: HasherRegistry) -> None:
    class MyStr(str):
        pass

    hasher = StrHasher("Name")
    assert hash_with(hasher, MyStr("a"), registry) == hash_with(hasher, "a", registry)


def test_str_hasher_hash_user_class_does_not_share_the_tag_of_a_builtin_name(
    registry: HasherRegistry,
) -> None:
    class int:  # noqa: A001, N801
        def __str__(self) -> str:
            return "1"

    assert hash_with(StrHasher(), int(), registry) != hash_with(StrHasher(), 1, registry)


def test_str_hasher_hash_user_classes_with_the_same_name_in_different_scopes_differ(
    registry: HasherRegistry,
) -> None:
    class Same:
        def __str__(self) -> str:
            return "x"

    def make() -> object:
        class Same:
            def __str__(self) -> str:
                return "x"

        return Same()

    assert hash_with(StrHasher(), Same(), registry) != hash_with(StrHasher(), make(), registry)


def test_str_hasher_encode_differs_from_the_repr_hasher() -> None:
    assert StrHasher().encode("a") != ReprHasher().encode("a")

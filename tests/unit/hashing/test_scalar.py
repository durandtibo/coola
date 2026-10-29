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
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available
from tests.unit.hashing.helpers import encode_leaf, hash_with, leaf_hash


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
            "0a465ca3e88a3a4c06a2865a8dcbe0ba8003136ee1e10c8039ea8af7580eb6ea",
            id="lowercase",
        ),
        pytest.param(
            "Hello",
            "7509515a347cc999c315cefa99ccc434079465380b8e38ff4b6bf0f688959a9c",
            id="uppercase_first",
        ),
        pytest.param(
            "hello ",
            "a85dd5ce008d668f8ddf73b72cd1bd97d2280276d181089a7f63f0a65e442629",
            id="trailing_space",
        ),
        pytest.param(
            "",
            "a09b15c53f4057a9a81fbcdd7fbcd67866dff35a5590aaaf5998ea5feee8c960",
            id="empty_string",
        ),
        pytest.param(
            "Meowwwwww",
            "918aa95e727394667b4587c1cfb39ff2f4e60d9a704452aadb2185642ae5fe95",
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
        pytest.param(16, "0a465ca3e88a3a4c", id="16"),
        pytest.param(32, "0a465ca3e88a3a4c06a2865a8dcbe0ba", id="32"),
        pytest.param(
            64, "0a465ca3e88a3a4c06a2865a8dcbe0ba8003136ee1e10c8039ea8af7580eb6ea", id="64-default"
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
            "59f582b5108d6c1c121e060256a236dc66e973a51ad009e89399a48a4ba66e34",
            id="lowercase",
        ),
        pytest.param(
            b"Hello",
            "699b52a85de5656967dbd1e52bb131e08a62fb5a8c80fc3364565e2e7c283d83",
            id="uppercase_first",
        ),
        pytest.param(
            b"hello ",
            "2e6c6814c0944afea8dc4652705ae941852f07767117d67f56e0fed2dfa1d085",
            id="trailing_space",
        ),
        pytest.param(
            b"",
            "ad75ef3400e3b77e9d8b11bcd744d0f358b90e1f277c86a999b92df785ddc4a5",
            id="empty_bytes",
        ),
        pytest.param(
            b"Meowwwwww",
            "0e2246fa0a141dae0327b967f86e1b49b055b481e84e4558b50caa32bbb55d93",
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
        pytest.param(16, "59f582b5108d6c1c", id="16"),
        pytest.param(32, "59f582b5108d6c1c121e060256a236dc", id="32"),
        pytest.param(
            64, "59f582b5108d6c1c121e060256a236dc66e973a51ad009e89399a48a4ba66e34", id="64-default"
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
    assert hash_with(EnumHasher(), Color.RED, registry) == leaf_hash(
        encode_leaf(EnumHasher(), Color.RED)
    )


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
    assert get_default_registry().hash(member) == leaf_hash(encode_leaf(EnumHasher(), member))


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
    assert get_default_registry().hash(member) == leaf_hash(encode_leaf(EnumHasher(), member))


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
            "11b80ab95ac066f82081baeeebdc67964e556c683ec2370bf3e6f783b5b2a647",
            id="int",
        ),
        pytest.param(
            0,
            "8067e713e20a64b38fe0c49c2d966cc9a1d6935545fb48194155d5e015043a5a",
            id="int_zero",
        ),
        pytest.param(
            -1,
            "e9d315b8de70e9500f6077d067adbec8fde34bd687533e1395aad7c1379ff855",
            id="int_negative",
        ),
        pytest.param(
            1.5,
            "bb7d0092c5be7ab08a61744098d71bf6cc7d553e98c19fa99703f9f1414c1784",
            id="float",
        ),
        pytest.param(
            float("inf"),
            "859c57ad4343ab200b395aee0621d85a796d458dae06f61a6d217e8f3bae173f",
            id="float_inf",
        ),
        pytest.param(
            True,
            "2dabca2e8bc28762bcdf8a5daeb58de61b4f26a7e7c06d9372cd328d124e62e0",
            id="bool_true",
        ),
        pytest.param(
            False,
            "4ccea996fe29ebaf6c6df31a94c2ccea4b6f905d664d11d321d677aa1f30f44a",
            id="bool_false",
        ),
        pytest.param(
            complex(1, 2),
            "06abc3468c3b9ea9e476e269e6e9fa1e168be15d98c38c6d3897a9ecace763c2",
            id="complex",
        ),
        pytest.param(
            complex(0, -1),
            "523c2e7026065be8e895c873643ca70244915f946cfdaa54d790ae05c9c38b63",
            id="complex_imaginary",
        ),
        pytest.param(
            None,
            "1e04f01fed015b7cf1213d1dd4139867935a82f34c292574722a64224042373c",
            id="none",
        ),
        pytest.param(
            "hello",
            "1d1bb12d476c02bfbf970f108bad8a0bfae8f523c48873dfb0111d5cc3933dc0",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "b7d99ea0d3967c203b50d9a701bbcf4e096066196876492aebddecfedd963c89",
            id="list",
        ),
        pytest.param(
            {"a": 1},
            "f2786b3350b432fa7e214b89caeb1dbf239840387431f23168aeef424dcde41a",
            id="dict",
        ),
        pytest.param(
            (1, 2, 3),
            "cef4265050afca2bca58c5515c87079a7364a437cd6c94a5b69e38c7bdf252a4",
            id="tuple",
        ),
        pytest.param(
            b"bytes",
            "bd255581a7f31f3e3d0f5fa83b08bf76cf2621be1145f2cf2c99ab8951e28902",
            id="bytes",
        ),
    ],
)
def test_repr_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert hash_with(ReprHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "11b80ab95ac066f8", id="16"),
        pytest.param(32, "11b80ab95ac066f82081baeeebdc6796", id="32"),
        pytest.param(
            64, "11b80ab95ac066f82081baeeebdc67964e556c683ec2370bf3e6f783b5b2a647", id="64-default"
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
            "69bcf1c0a109c54d2781a94c903016e096419a61760c8a32e8e8b1d65a20a89d",
            id="int",
        ),
        pytest.param(
            0,
            "98a04ad3ba306491b28b5abbc830c49a12754a2b057033dd774e969dc774a275",
            id="int_zero",
        ),
        pytest.param(
            -1,
            "cff2c7dfaf05159186b2b32fb8fbbe84d517af38b3bc258df02489e4455a100b",
            id="int_negative",
        ),
        pytest.param(
            1.5,
            "2f06dc65964f2412520a9267b23416ee27c05dedc8286bcb86e537ba8769caab",
            id="float",
        ),
        pytest.param(
            float("inf"),
            "7a11526692ccbc572863cfff60fe708135ef588494ff1566600cc2e76674e64d",
            id="float_inf",
        ),
        pytest.param(
            True,
            "f79be2da3f4ea45034f6a8260340ec0a3e45ab1a96407000e8086b29fa0e3660",
            id="bool_true",
        ),
        pytest.param(
            False,
            "034410805673cda59a26936f781d4e5a92a058aabd578b7b9b15045c256101fe",
            id="bool_false",
        ),
        pytest.param(
            complex(1, 2),
            "3897d4641540831678ab372f277bd8f3bccbd16e6cf40d00270ededd4d3da4c0",
            id="complex",
        ),
        pytest.param(
            complex(0, -1),
            "39c969c599430ff093c02879810b3c68eae18c38579cc0a22fa22a4a7de09fe7",
            id="complex_imaginary",
        ),
        pytest.param(
            None,
            "2118c9881ae2b84ac507a0fbd200c212dbefb68a93a5c2c88aa802b249c19a6f",
            id="none",
        ),
        pytest.param(
            "hello",
            "e806f3c0c67a48f80d326113f3f99dced865dc42ac62ff87da0489b6bdb13ea4",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "42edfff29d19c1f109fb6b4b9a2b578c74599328a4d5001491f1a5770f38026f",
            id="list",
        ),
        pytest.param(
            {"a": 1},
            "767d45f75038f0ce27a51595541794d0d433308a297b484b40c536cbac90c402",
            id="dict",
        ),
        pytest.param(
            (1, 2, 3),
            "6eaeaef2e269c69af5a93bd4011bff0c5946394eb0ac070b3858b07b28f571e6",
            id="tuple",
        ),
        pytest.param(
            b"bytes",
            "a4af9f2b69d21c85b54f7900f1dcafc0493b6750d20ce8256e9ca08ee8951629",
            id="bytes",
        ),
    ],
)
def test_str_hasher_hash_parametrized(data: Any, expected: str, registry: HasherRegistry) -> None:
    assert hash_with(StrHasher(), data, registry) == expected


@pytest.mark.parametrize(
    ("length", "expected"),
    [
        pytest.param(16, "e806f3c0c67a48f8", id="16"),
        pytest.param(32, "e806f3c0c67a48f80d326113f3f99dce", id="32"),
        pytest.param(
            64, "e806f3c0c67a48f80d326113f3f99dced865dc42ac62ff87da0489b6bdb13ea4", id="64-default"
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

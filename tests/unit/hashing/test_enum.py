from __future__ import annotations

from enum import Enum, Flag, IntEnum, StrEnum

import pytest

from coola.hashing import EnumHasher, HasherRegistry, get_default_registry
from tests.unit.hashing.helpers import hash_with, leaf_hash


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


class MixedStr(str, Enum):  # noqa: UP042
    A = "a"


class MixedInt(int, Enum):
    ONE = 1


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry()


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

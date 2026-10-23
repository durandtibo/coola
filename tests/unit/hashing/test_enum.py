from __future__ import annotations

from enum import Enum, Flag, IntEnum, StrEnum

import pytest

from coola.hashing import EnumHasher, HasherRegistry, get_default_registry, hash_enum


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


def test_enum_hasher_hash_matches_hash_enum(registry: HasherRegistry) -> None:
    assert EnumHasher().hash(Color.RED, registry=registry) == hash_enum(Color.RED)


@pytest.mark.parametrize("length", [2, 16, 64, 128])
def test_enum_hasher_hash_length(length: int, registry: HasherRegistry) -> None:
    assert len(EnumHasher().hash(Color.RED, registry=registry, length=length)) == length


def test_enum_hasher_hash_invalid_length_raises(registry: HasherRegistry) -> None:
    with pytest.raises(ValueError, match="length must be an even number"):
        EnumHasher().hash(Color.RED, registry=registry, length=3)


###############################
#     Tests for hash_enum     #
###############################


def test_hash_enum_different_members_differ() -> None:
    assert hash_enum(Color.RED) != hash_enum(Color.BLUE)


def test_hash_enum_same_member_name_in_different_classes_differ() -> None:
    assert hash_enum(Color.RED) != hash_enum(Shade.RED)


def test_hash_enum_is_deterministic() -> None:
    assert hash_enum(Color.RED) == hash_enum(Color.RED)


def test_hash_enum_composite_flag() -> None:
    assert hash_enum(Perm.R | Perm.W) != hash_enum(Perm.R)


###################################################
#     Tests for enums in the default registry     #
###################################################


@pytest.mark.parametrize("member", [Color.RED, Number.ONE, Label.A, MixedStr.A, MixedInt.ONE])
def test_default_registry_hashes_enum_members_with_hash_enum(member: Enum) -> None:
    assert get_default_registry().hash(member) == hash_enum(member)


def test_default_registry_enum_is_not_collapsed_into_value() -> None:
    registry = get_default_registry()
    assert registry.hash(Number.ONE) != registry.hash(1)
    assert registry.hash(Label.A) != registry.hash("a")
    assert registry.hash(MixedStr.A) != registry.hash("a")
    assert registry.hash(MixedInt.ONE) != registry.hash(1)


class MixedBytes(bytes, Enum):
    A = b"a"


class MixedTuple(tuple, Enum):
    A = (1, 2)


@pytest.mark.parametrize("member", [MixedBytes.A, MixedTuple.A])
def test_default_registry_hashes_enum_mixins_as_enum(member: Enum) -> None:
    registry = get_default_registry()
    assert registry.hash(member) == hash_enum(member)
    assert registry.hash(member) != registry.hash(member.value)

from __future__ import annotations

import uuid
import warnings
from collections import UserString
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest

from coola.hashing import (
    BytesHasher,
    DataclassHasher,
    DatetimeHasher,
    HasherRegistry,
    MappingHasher,
    PathHasher,
    ReprHasher,
    SequenceHasher,
    StrHasher,
    StringHasher,
    get_default_registry,
    get_stable_registry,
    hash_object,
    interface as hashing_interface,
    register_hashers,
    tagged,
)
from coola.hashing.interface import _register_default_hashers
from coola.hashing.pydantic import PydanticModelHasher
from coola.registry import AlreadyRegisteredError, TypeNotRegisteredError
from coola.testing.fixtures import numpy_available, pydantic_available
from coola.utils.imports import import_object, is_numpy_available, is_pydantic_available
from tests.unit.hashing.helpers import Unhashable, leaf_hash

if is_numpy_available():
    import numpy as np

if is_pydantic_available():
    import pydantic


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the singleton registry before and after each test."""
    hashing_interface._default_registry.reset()
    yield
    hashing_interface._default_registry.reset()


class CustomList(list):
    r"""Custom class that inherits from list."""


#################################
#     Tests for hash_object     #
#################################


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            5,
            "febcf8e6dee75c90080cab5135bdd77b752cf87dd2f9bc39746793fb67522c4b",
            id="int",
        ),
        pytest.param(
            3.14,
            "9855afb41eaf84538d763ca01faec06219f2bf9cd43a92ad6ecee96b340af804",
            id="float",
        ),
        pytest.param(
            complex(1, 2),
            "475b70f206b62fc93989cfc073a63c05b23accc90b082d77d984f4c3c87b1862",
            id="complex",
        ),
        pytest.param(
            True,
            "b125316ff709ad952976c214852acc80ef2e75dd12dd50a908a204705d373820",
            id="bool",
        ),
        pytest.param(
            "hello",
            "2998e64e4b20ac91774f304f5527a5160d30249cdcdc984effac931cf54196e0",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "824284b9d2ee9802ca0d8a94a16de30c1078fe55a55ee9b74943681fbe9147dd",
            id="list",
        ),
        pytest.param(
            (1, 2, 3),
            "039d1e5e77772efd4a39b67bfc3a51b9c51586626d7c860f87951462d9faf0b8",
            id="tuple",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            "e5a0bcc2799f418c155af1493bf8e86f270d5d61e002400b5cf8b79facb1bc88",
            id="dict",
        ),
        pytest.param(
            date(2021, 1, 1),
            "5f04f6be9602d7a5af0207e8b221a3bceb87050c6ec80f5f818dad202bf76c5f",
            id="date",
        ),
        pytest.param(
            datetime(2021, 1, 1, tzinfo=UTC),
            "9def71436f2f37949ee5c665e24af39f9886cf3e919baacdc81c77ba06493805",
            id="datetime",
        ),
    ],
)
def test_hash_object_parametrized(data: object, expected: str) -> None:
    assert hash_object(data) == expected


def test_hash_object_returns_str() -> None:
    assert isinstance(hash_object(42), str)


def test_hash_object_is_deterministic() -> None:
    assert hash_object([1, 2, 3]) == hash_object([1, 2, 3])


def test_hash_object_with_custom_registry() -> None:
    # A registry with only ReprHasher treats lists as leaf values.
    registry = HasherRegistry({object: ReprHasher()})
    assert hash_object([1, 2, 3], registry=registry) == leaf_hash(tagged("list", repr([1, 2, 3])))


def test_hash_object_with_custom_length_returns_str() -> None:
    result = hash_object("hello", length=16)
    assert isinstance(result, str)


def test_hash_object_with_custom_length_length() -> None:
    result = hash_object("hello", length=16)
    assert len(result) == 16


def test_hash_object_uses_default_registry_when_none() -> None:
    assert hash_object([1, 2, 3]) == hash_object([1, 2, 3], registry=get_default_registry())


def test_hash_object_nested_structure() -> None:
    assert hash_object({"a": [1, 2], "b": [3, 4]}) == (
        "0aee2e48d5414fb43c8e14f3ab931a9622a28565f9804e5c54310eac7ab4e44a"
    )


def test_hash_object_unhashable_raises_by_default() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_object(Unhashable())


def test_hash_object_unhashable_raises_when_explicitly_false() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_object(Unhashable(), ignore_unhashable=False)


def test_hash_object_ignore_unhashable_returns_str_warns() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        hash_object(Unhashable(), ignore_unhashable=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hash_object_ignore_unhashable_returns_str() -> None:
    assert isinstance(hash_object(Unhashable(), ignore_unhashable=True), str)


def test_hash_object_ignore_unhashable_nested_in_list_warns() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        hash_object([1, Unhashable(), 3], ignore_unhashable=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hash_object_ignore_unhashable_nested_in_list() -> None:
    assert isinstance(hash_object([1, Unhashable(), 3], ignore_unhashable=True), str)


def test_hash_object_ignore_unhashable_nested_in_dict_warns() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        hash_object({"a": Unhashable()}, ignore_unhashable=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hash_object_ignore_unhashable_nested_in_dict() -> None:
    assert isinstance(hash_object({"a": Unhashable()}, ignore_unhashable=True), str)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hash_object_ignore_unhashable_is_deterministic() -> None:
    assert hash_object(Unhashable(), ignore_unhashable=True) == hash_object(
        Unhashable(), ignore_unhashable=True
    )


######################################
#     Tests for register_hashers     #
######################################


def test_register_hashers_adds_to_default_registry() -> None:
    register_hashers({CustomList: SequenceHasher()})
    assert isinstance(get_default_registry().find_hasher(CustomList), SequenceHasher)


def test_register_hashers_with_exist_ok_true() -> None:
    register_hashers({CustomList: ReprHasher()})
    register_hashers({CustomList: SequenceHasher()}, exist_ok=True)


def test_register_hashers_with_exist_ok_false() -> None:
    register_hashers({CustomList: ReprHasher()})
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        register_hashers({CustomList: SequenceHasher()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    assert isinstance(get_default_registry(), HasherRegistry)


def test_get_default_registry_returns_singleton() -> None:
    assert get_default_registry() is get_default_registry()


def test_get_default_registry_singleton_persists_modifications() -> None:
    registry1 = get_default_registry()
    registry1.register(CustomList, SequenceHasher())
    assert isinstance(registry1.find_hasher(CustomList), SequenceHasher)


def test_get_default_registry_singleton_returns_the_same_registry() -> None:
    registry1 = get_default_registry()
    registry1.register(CustomList, SequenceHasher())
    assert registry1 is get_default_registry()


def test_get_default_registry_singleton_persists_modifications_across_calls() -> None:
    get_default_registry().register(CustomList, SequenceHasher())
    assert isinstance(get_default_registry().find_hasher(CustomList), SequenceHasher)


def test_get_default_registry_does_not_register_object() -> None:
    # object is no longer registered as a catch-all fallback.
    assert not get_default_registry().has_hasher(object)


def test_get_default_registry_registers_numeric_types_bool() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(bool), ReprHasher)


def test_get_default_registry_registers_numeric_types_int() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(int), ReprHasher)


def test_get_default_registry_registers_numeric_types_float() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(float), ReprHasher)


def test_get_default_registry_registers_numeric_types_complex() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(complex), ReprHasher)


def test_get_default_registry_registers_str() -> None:
    assert isinstance(get_default_registry().find_hasher(str), StringHasher)


def test_get_default_registry_registers_bytes() -> None:
    assert isinstance(get_default_registry().find_hasher(bytes), BytesHasher)


def test_get_default_registry_registers_sequences_list() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(list), SequenceHasher)


def test_get_default_registry_registers_sequences_tuple() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(tuple), SequenceHasher)


def test_get_default_registry_registers_sequences_sequence() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(Sequence), SequenceHasher)


def test_get_default_registry_registers_mappings_dict() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(dict), MappingHasher)


def test_get_default_registry_registers_mappings_mapping() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(Mapping), MappingHasher)


def test_get_default_registry_registers_datetime_types_date() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(date), DatetimeHasher)


def test_get_default_registry_registers_datetime_types_datetime() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(datetime), DatetimeHasher)


def test_get_default_registry_registers_path_types() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(Path), PathHasher)


def test_get_default_registry_registers_none() -> None:
    assert isinstance(get_default_registry().find_hasher(type(None)), ReprHasher)


def test_get_default_registry_can_hash_list() -> None:
    assert get_default_registry().hash([1, 2, 3]) == (
        "824284b9d2ee9802ca0d8a94a16de30c1078fe55a55ee9b74943681fbe9147dd"
    )


def test_get_default_registry_can_hash_dict() -> None:
    assert get_default_registry().hash({"a": 1, "b": 2}) == (
        "e5a0bcc2799f418c155af1493bf8e86f270d5d61e002400b5cf8b79facb1bc88"
    )


@pydantic_available
def test_register_default_hashers_registers_pydantic_model_when_available() -> None:
    registry = HasherRegistry()
    _register_default_hashers(registry)
    assert isinstance(registry.find_hasher(pydantic.BaseModel), PydanticModelHasher)


@pydantic_available
@patch("coola.hashing.interface.package_available", lambda name: False)  # noqa: ARG005
def test_register_default_hashers_skips_pydantic_model_when_unavailable() -> None:
    registry = HasherRegistry()
    _register_default_hashers(registry)
    assert not registry.has_hasher(pydantic.BaseModel)


@patch("coola.hashing.interface.package_available", lambda name: False)  # noqa: ARG005
def test_register_default_hashers_registers_the_other_hashers_when_pydantic_is_unavailable() -> (
    None
):
    registry = HasherRegistry()
    _register_default_hashers(registry)
    assert registry.has_hasher(list)


######################################################
#     Tests for the positional arguments (legacy)    #
######################################################


def test_hash_object_keyword_arguments_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hash_object([1], registry=HasherRegistry({list: SequenceHasher(), int: ReprHasher()}))
        hash_object([1], length=16, ignore_unhashable=True)


def test_hash_object_positional_registry_is_deprecated() -> None:
    registry = get_default_registry()
    with pytest.warns(DeprecationWarning, match="positional arguments is deprecated"):
        hash_object([1], registry)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_hash_object_positional_registry_is_deprecated_result() -> None:
    registry = get_default_registry()
    result = hash_object([1], registry)
    assert result == hash_object([1], registry=registry)


def test_hash_object_positional_registry_none_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="positional arguments is deprecated"):
        hash_object([1], None)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_hash_object_positional_registry_none_is_deprecated_result() -> None:
    result = hash_object([1], None)
    assert result == hash_object([1])


def test_hash_object_positional_length_and_ignore_unhashable_are_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="positional arguments is deprecated"):
        hash_object([1], None, 16, False)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_hash_object_positional_length_and_ignore_unhashable_are_deprecated_result() -> None:
    result = hash_object([1], None, 16, False)
    assert result == hash_object([1], length=16)


def test_hash_object_positional_warning_points_to_the_caller() -> None:
    with pytest.warns(DeprecationWarning, match="is deprecated") as record:
        hash_object([1], None)
    assert record[0].filename == __file__


def test_hash_object_positional_length_instead_of_registry_raises() -> None:
    with pytest.raises(TypeError, match="`registry`, which must be a HasherRegistry or None"):
        hash_object([1], 32)  # type: ignore[call-overload]


def test_hash_object_too_many_positional_arguments_raises() -> None:
    with pytest.raises(TypeError, match="takes at most 4 positional arguments but 5 were given"):
        hash_object([1], None, 16, False, 1)  # type: ignore[call-overload]


@pytest.mark.parametrize(
    ("args", "kwargs", "name"),
    [
        ((None, 16), {"length": 32}, "length"),
        ((None, 16, True), {"ignore_unhashable": True}, "ignore_unhashable"),
        ((HasherRegistry(),), {"registry": HasherRegistry()}, "registry"),
    ],
)
def test_hash_object_argument_given_by_position_and_keyword_raises(
    args: tuple, kwargs: dict, name: str
) -> None:
    with (
        pytest.warns(DeprecationWarning, match="is deprecated"),
        pytest.raises(TypeError, match=f"multiple values for argument '{name}'"),
    ):
        hash_object([1], *args, **kwargs)


################################################
#     Tests for the types of the registry      #
################################################


@dataclass
class _Point:
    x: int
    y: int


@dataclass
class _Line:
    start: _Point
    end: _Point


class _HasContent:
    def __init__(self, content: object = (1, 2)) -> None:
        self.content = content

    def __coola_hash__(self) -> object:
        return self.content


def test_hash_object_dataclass_same_value() -> None:
    assert hash_object(_Point(1, 2)) == hash_object(_Point(1, 2))


def test_hash_object_dataclass_different_value() -> None:
    assert hash_object(_Point(1, 2)) != hash_object(_Point(2, 1))


def test_hash_object_nested_dataclass_same_value() -> None:
    line = _Line(_Point(0, 0), _Point(1, 1))
    assert hash_object(line) == hash_object(_Line(_Point(0, 0), _Point(1, 1)))


def test_hash_object_nested_dataclass_different_value() -> None:
    line = _Line(_Point(0, 0), _Point(1, 1))
    assert hash_object(line) != hash_object(_Line(_Point(0, 0), _Point(1, 2)))


def test_hash_object_nested_dataclass_in_containers() -> None:
    line = _Line(_Point(0, 0), _Point(1, 1))
    assert hash_object([line, {"k": line}])


def test_hash_object_dataclass_is_tagged_with_the_class() -> None:
    @dataclass
    class Other:
        x: int
        y: int

    assert hash_object(_Point(1, 2)) != hash_object(Other(1, 2))


def test_hash_object_dataclass_class_is_not_an_instance() -> None:
    with pytest.raises(KeyError, match="is not registered"):
        hash_object(_Point)


def test_hash_object_dataclass_registered_hasher_wins_registered_hasher() -> None:
    registry = HasherRegistry()
    _register_default_hashers(registry)
    hasher = StrHasher()
    registry.register(_Line, hasher)
    assert registry.find_hasher(_Line) is hasher


def test_hash_object_dataclass_registered_hasher_wins_default_hasher() -> None:
    registry = HasherRegistry()
    _register_default_hashers(registry)
    hasher = StrHasher()
    registry.register(_Line, hasher)
    assert isinstance(registry.find_hasher(_Point), DataclassHasher)


def test_hash_object_custom_object_equal_content_same_hash() -> None:
    assert hash_object(_HasContent((1, 2))) == hash_object(_HasContent((1, 2)))


def test_hash_object_custom_object_different_content_different_hash() -> None:
    assert hash_object(_HasContent((1, 2))) != hash_object(_HasContent((1, 3)))


def test_hash_object_custom_object_differs_from_its_content() -> None:
    assert hash_object(_HasContent((1, 2))) != hash_object((1, 2))


def test_hash_object_custom_object_differs_from_another_class_with_the_same_content() -> None:
    class Other(_HasContent):
        pass

    assert hash_object(_HasContent((1, 2))) != hash_object(Other((1, 2)))


def test_hash_object_custom_object_content_is_hashed_through_the_registry() -> None:
    registry = get_default_registry().copy()
    registry.register(str, ReprHasher(), exist_ok=True)
    assert hash_object(_HasContent("a"), registry=registry) != hash_object(_HasContent("a"))


def test_hash_object_custom_object_in_a_list() -> None:
    assert hash_object([_HasContent((1, 2))]) == hash_object([_HasContent((1, 2))])


def test_hash_object_custom_object_that_contains_itself_raises() -> None:
    obj = _HasContent()
    obj.content = [obj]
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(obj)


def test_hash_object_dataclass_with_coola_hash_method_uses_the_dataclass_hasher() -> None:
    @dataclass
    class Both:
        x: int

        def __coola_hash__(self) -> object:
            return "constant"

    assert hash_object(Both(1)) != hash_object(Both(2))


def test_hash_object_unregistered_type_raises() -> None:
    class Plain: ...

    with pytest.raises(KeyError, match="is not registered"):
        hash_object(Plain())


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(UUID(int=1), id="uuid"),
        pytest.param(Decimal("1.5"), id="decimal"),
        pytest.param(Fraction(1, 3), id="fraction"),
        pytest.param(range(3), id="range"),
        pytest.param(ZoneInfo("UTC"), id="zoneinfo"),
    ],
)
def test_hash_object_standard_library_values_length(data: object) -> None:
    assert len(hash_object(data)) == 64


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(UUID(int=1), id="uuid"),
        pytest.param(Decimal("1.5"), id="decimal"),
        pytest.param(Fraction(1, 3), id="fraction"),
        pytest.param(range(3), id="range"),
        pytest.param(ZoneInfo("UTC"), id="zoneinfo"),
    ],
)
def test_hash_object_standard_library_values_deterministic(data: object) -> None:
    assert hash_object(data) == hash_object(data)


def test_hash_object_standard_library_values_are_different_uuid() -> None:
    assert hash_object(UUID(int=1)) != hash_object(UUID(int=2))


def test_hash_object_standard_library_values_are_different_uuid_and_str() -> None:
    assert hash_object(UUID(int=1)) != hash_object(str(UUID(int=1)))


def test_hash_object_standard_library_values_are_different_zone_info() -> None:
    assert hash_object(ZoneInfo("UTC")) != hash_object(ZoneInfo("Europe/Paris"))


def test_hash_object_standard_library_values_are_different_decimal() -> None:
    assert hash_object(Decimal(1)) != hash_object(1)


def test_hash_object_standard_library_values_are_different_fraction() -> None:
    assert hash_object(Fraction(1, 1)) != hash_object(1)


def test_hash_object_standard_library_values_are_different_range() -> None:
    assert hash_object(range(3)) != hash_object([0, 1, 2])


def test_hash_object_equal_values_have_the_same_hash_decimal() -> None:
    assert hash_object(Decimal("1.0")) == hash_object(Decimal("1.00"))


def test_hash_object_equal_values_have_the_same_hash_fraction() -> None:
    assert hash_object(Fraction(1, 2)) == hash_object(Fraction(2, 4))


def test_hash_object_equal_values_have_the_same_hash_range() -> None:
    assert hash_object(range(0)) == hash_object(range(5, 5))


def test_hash_object_equal_values_have_the_same_hash_nested() -> None:
    assert hash_object({"k": [Decimal("1.0")]}) == hash_object({"k": [Decimal(1)]})


@numpy_available
@pytest.mark.parametrize(
    ("scalar", "builtin"),
    [
        pytest.param("float32", 1.5, id="float32"),
        pytest.param("float64", 1.5, id="float64"),
        pytest.param("float16", 1.5, id="float16"),
        pytest.param("int8", 3, id="int8"),
        pytest.param("int64", 3, id="int64"),
        pytest.param("uint16", 3, id="uint16"),
        pytest.param("bool_", True, id="bool"),
        pytest.param("complex64", 1 + 2j, id="complex64"),
        pytest.param("complex128", 1 + 2j, id="complex128"),
    ],
)
def test_hash_object_numpy_scalar_hashes_like_the_builtin_value(
    scalar: str, builtin: object
) -> None:
    assert hash_object(getattr(np, scalar)(builtin)) == hash_object(builtin)


@numpy_available
def test_hash_object_numpy_scalar_in_a_container() -> None:
    assert hash_object({"a": [np.float32(0.5), np.int64(2)]}) == hash_object({"a": [0.5, 2]})


@numpy_available
def test_hash_object_numpy_scalar_different_values_float32() -> None:
    assert hash_object(np.float32(1.0)) != hash_object(np.float32(2.0))


@numpy_available
def test_hash_object_numpy_scalar_different_values_int64_and_float64() -> None:
    assert hash_object(np.int64(1)) != hash_object(np.float64(1.0))


@numpy_available
def test_hash_object_numpy_scalar_different_values_bool() -> None:
    assert hash_object(np.bool_(True)) != hash_object(1)


def test_hash_object_invalid_length_is_rejected_before_hashing() -> None:
    # The unhashable value would raise another error if the data was traversed first.
    with pytest.raises(ValueError, match="even number"):
        hash_object([Unhashable()], length=3)


def test_hash_object_memoryview_differs_from_bytes() -> None:
    assert hash_object(memoryview(b"abc")) != hash_object(b"abc")


def test_hash_object_memoryview_differs_from_bytearray() -> None:
    assert hash_object(memoryview(b"abc")) != hash_object(bytearray(b"abc"))


def test_hash_object_equal_memoryviews_have_the_same_hash() -> None:
    assert hash_object(memoryview(b"abc")) == hash_object(memoryview(b"abc"))


def test_hash_object_user_string_is_hashable() -> None:
    assert len(hash_object(UserString("abc"))) == 64


def test_hash_object_equal_user_strings_have_the_same_hash() -> None:
    assert hash_object(UserString("abc")) == hash_object(UserString("abc"))


def test_hash_object_different_user_strings_have_different_hashes() -> None:
    assert hash_object(UserString("abc")) != hash_object(UserString("abd"))


def test_hash_object_user_string_differs_from_str() -> None:
    assert hash_object(UserString("abc")) != hash_object("abc")


def test_hash_object_user_string_in_a_list() -> None:
    assert hash_object([UserString("abc")]) != hash_object(["abc"])


def test_hash_object_mapping_proxy_is_hashable() -> None:
    assert len(hash_object(MappingProxyType({"a": 1}))) == 64


def test_hash_object_equal_mapping_proxies_have_the_same_hash() -> None:
    assert hash_object(MappingProxyType({"a": 1, "b": 2})) == hash_object(
        MappingProxyType({"b": 2, "a": 1})
    )


def test_hash_object_different_mapping_proxies_have_different_hashes() -> None:
    assert hash_object(MappingProxyType({"a": 1})) != hash_object(MappingProxyType({"a": 2}))


def test_hash_object_mapping_proxy_differs_from_dict() -> None:
    assert hash_object(MappingProxyType({"a": 1})) != hash_object({"a": 1})


def test_hash_object_user_class_named_uuid_does_not_collide_with_uuid() -> None:
    class UUID:
        def __str__(self) -> str:
            return str(uuid.UUID(int=1))

    registry = get_default_registry().copy()
    registry.register(UUID, StrHasher())
    assert registry.hash(UUID()) != registry.hash(uuid.UUID(int=1))


def test_hash_object_uuid_subclass_hashes_like_uuid() -> None:
    class MyUUID(uuid.UUID):
        pass

    assert hash_object(MyUUID(int=1)) == hash_object(uuid.UUID(int=1))


def test_hash_object_datetime_subclass_hashes_like_datetime() -> None:
    class MyDatetime(datetime):
        pass

    assert hash_object(MyDatetime(2024, 1, 2, tzinfo=UTC)) == hash_object(
        datetime(2024, 1, 2, tzinfo=UTC)
    )


class _OtherHash:
    def hash(self, seed: int = 0) -> int:
        return seed


def test_hash_object_hash_method_with_another_signature_raises() -> None:
    with pytest.raises(TypeNotRegisteredError, match="register a hasher"):
        hash_object(_OtherHash())


def test_hash_object_hash_method_with_another_signature_ignore_unhashable() -> None:
    with pytest.warns(UserWarning, match="No hasher registered"):
        hash_object(_OtherHash(), ignore_unhashable=True)


#############################################
#     Tests for the lazy hashers     #
#############################################

LAZY_MODULES = [
    pytest.param("coola.hashing.jax", ("jax", "jaxlib"), id="jax"),
    pytest.param("coola.hashing.numpy", ("numpy",), id="numpy"),
    pytest.param("coola.hashing.pandas", ("pandas",), id="pandas"),
    pytest.param("coola.hashing.polars", ("polars",), id="polars"),
    pytest.param("coola.hashing.pyarrow", ("pyarrow",), id="pyarrow"),
    pytest.param("coola.hashing.torch", ("torch",), id="torch"),
    pytest.param("coola.hashing.xarray", ("xarray",), id="xarray"),
    pytest.param("coola.hashing.pydantic", ("pydantic",), id="pydantic"),
]

LAZY_MODULE_NAMES = [
    "coola.hashing.jax",
    "coola.hashing.numpy",
    "coola.hashing.pandas",
    "coola.hashing.polars",
    "coola.hashing.pyarrow",
    "coola.hashing.torch",
    "coola.hashing.xarray",
    "coola.hashing.pydantic",
]


@pytest.mark.parametrize(("module", "names"), LAZY_MODULES)
def test_get_default_hashers_types_come_from_the_library(
    module: str, names: tuple[str, ...]
) -> None:
    """Test that the lazy group is loaded for each type that it
    registers."""
    pytest.importorskip(names[0])
    mapping = import_object(f"{module}:get_default_hashers")()
    assert {t.__module__.partition(".")[0] for t in mapping} <= set(names)


@pytest.mark.parametrize(("module", "names"), LAZY_MODULES)
def test_get_default_hashers_returns_entries_if_the_library_is_available(
    module: str, names: tuple[str, ...]
) -> None:
    pytest.importorskip(names[0])
    assert len(import_object(f"{module}:get_default_hashers")()) > 0


@pytest.mark.parametrize("module", LAZY_MODULE_NAMES)
def test_get_default_hashers_returns_nothing_if_the_library_is_unavailable(module: str) -> None:
    name = module.rpartition(".")[2]
    with patch(f"{module}.is_{name}_available", lambda: False):
        assert import_object(f"{module}:get_default_hashers")() == {}


##########################################
#     Tests for get_stable_registry      #
##########################################


def test_get_stable_registry_returns_registry() -> None:
    assert isinstance(get_stable_registry(), HasherRegistry)


def test_get_stable_registry_returns_singleton() -> None:
    assert get_stable_registry() is get_stable_registry()


def test_get_stable_registry_is_not_the_default_registry() -> None:
    assert get_stable_registry() is not get_default_registry()


def test_get_stable_registry_has_the_builtin_hashers() -> None:
    assert get_stable_registry().has_hasher(str)


def test_get_stable_registry_is_not_changed_by_register_hashers() -> None:
    class MyType:
        pass

    register_hashers({MyType: StrHasher()})
    assert not get_stable_registry().has_hasher(MyType)

from __future__ import annotations

import warnings
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
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
    hash_object,
    hash_tagged,
    interface as hashing_interface,
    register_hashers,
)
from coola.hashing.interface import _register_default_hashers
from coola.hashing.pydantic import PydanticModelHasher
from coola.registry import AlreadyRegisteredError, TypeNotRegisteredError
from coola.testing.fixtures import numpy_available, polars_available, pydantic_available
from coola.utils.imports import is_pydantic_available
from tests.unit.hashing.helpers import Unhashable

if is_pydantic_available():
    import pydantic


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the singleton registry before and after each test."""
    hashing_interface._default_registry._instance = None
    yield
    hashing_interface._default_registry._instance = None


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
            "e082b7ec382aca2fd735131e1e0783773d3024db6b45dcc12663f1f58ee2acbe",
            id="int",
        ),
        pytest.param(
            3.14,
            "7f490c17244f4f05739d330c6b390f4d6ca1a259c63527d49ed4031de9f80fca",
            id="float",
        ),
        pytest.param(
            complex(1, 2),
            "4ee227f701d6a0eaa2429bfb3378dd865abe0e5fc9d303068c0befaffbab41b1",
            id="complex",
        ),
        pytest.param(
            True,
            "d7d4d5d42aa13a2cb19b77506ff0e2cfe278713fecddd620b898d65bc7dc63dd",
            id="bool",
        ),
        pytest.param(
            "hello",
            "9d4bd2af5629d5a22405d7fe02fe410ee11ae44178b4ee0eb9ca4d7795ac9906",
            id="str",
        ),
        pytest.param(
            [1, 2, 3],
            "ec609b5623c6456c63f338d63d699ebe4d5345baff999c5fc61471c62656f586",
            id="list",
        ),
        pytest.param(
            (1, 2, 3),
            "68c32ffc6288f744784a055a2abd7392488a36623521e708f52da350c9038f40",
            id="tuple",
        ),
        pytest.param(
            {"a": 1, "b": 2},
            "cefc870254126d81d0209ed1b37eda8d4d9c06378205a1c802dd2a05b21dc896",
            id="dict",
        ),
        pytest.param(
            date(2021, 1, 1),
            "1d7ccf2f2d2fa168388daafdef0be5efb6e74b780c83a649af3368a5a23911f1",
            id="date",
        ),
        pytest.param(
            datetime(2021, 1, 1, tzinfo=UTC),
            "1cc1275c7d33a7dc33384372f2b37341544cd099de233c9bcaa753fe08ea6621",
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
    assert hash_object([1, 2, 3], registry=registry) == hash_tagged("list", repr([1, 2, 3]))


def test_hash_object_with_custom_length() -> None:
    result = hash_object("hello", length=16)
    assert isinstance(result, str)
    assert len(result) == 16


def test_hash_object_uses_default_registry_when_none() -> None:
    assert hash_object([1, 2, 3]) == hash_object([1, 2, 3], registry=get_default_registry())


def test_hash_object_nested_structure() -> None:
    assert hash_object({"a": [1, 2], "b": [3, 4]}) == (
        "df457ece3d6ba8364e163149129d84e5dad821264bc2fbaa5d239b07aa14be37"
    )


def test_hash_object_unhashable_raises_by_default() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_object(Unhashable())


def test_hash_object_unhashable_raises_when_explicitly_false() -> None:
    with pytest.raises(KeyError, match=r"is not registered"):
        hash_object(Unhashable(), ignore_unhashable=False)


def test_hash_object_ignore_unhashable_returns_str() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        assert isinstance(hash_object(Unhashable(), ignore_unhashable=True), str)


def test_hash_object_ignore_unhashable_nested_in_list() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        assert isinstance(hash_object([1, Unhashable(), 3], ignore_unhashable=True), str)


def test_hash_object_ignore_unhashable_nested_in_dict() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
        assert isinstance(hash_object({"a": Unhashable()}, ignore_unhashable=True), str)


def test_hash_object_ignore_unhashable_is_deterministic() -> None:
    with pytest.warns(UserWarning, match=r"No hasher registered"):
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

    registry2 = get_default_registry()
    assert registry1 is registry2
    assert isinstance(registry2.find_hasher(CustomList), SequenceHasher)


def test_get_default_registry_does_not_register_object() -> None:
    # object is no longer registered as a catch-all fallback.
    assert not get_default_registry().has_hasher(object)


def test_get_default_registry_registers_numeric_types() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(bool), ReprHasher)
    assert isinstance(registry.find_hasher(int), ReprHasher)
    assert isinstance(registry.find_hasher(float), ReprHasher)
    assert isinstance(registry.find_hasher(complex), ReprHasher)


def test_get_default_registry_registers_str() -> None:
    assert isinstance(get_default_registry().find_hasher(str), StringHasher)


def test_get_default_registry_registers_bytes() -> None:
    assert isinstance(get_default_registry().find_hasher(bytes), BytesHasher)


def test_get_default_registry_registers_sequences() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(list), SequenceHasher)
    assert isinstance(registry.find_hasher(tuple), SequenceHasher)
    assert isinstance(registry.find_hasher(Sequence), SequenceHasher)


def test_get_default_registry_registers_mappings() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(dict), MappingHasher)
    assert isinstance(registry.find_hasher(Mapping), MappingHasher)


def test_get_default_registry_registers_datetime_types() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(date), DatetimeHasher)
    assert isinstance(registry.find_hasher(datetime), DatetimeHasher)


def test_get_default_registry_registers_path_types() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_hasher(Path), PathHasher)


def test_get_default_registry_registers_none() -> None:
    assert isinstance(get_default_registry().find_hasher(type(None)), ReprHasher)


def test_get_default_registry_can_hash_list() -> None:
    assert get_default_registry().hash([1, 2, 3]) == (
        "ec609b5623c6456c63f338d63d699ebe4d5345baff999c5fc61471c62656f586"
    )


def test_get_default_registry_can_hash_dict() -> None:
    assert get_default_registry().hash({"a": 1, "b": 2}) == (
        "cefc870254126d81d0209ed1b37eda8d4d9c06378205a1c802dd2a05b21dc896"
    )


@pydantic_available
def test_register_default_hashers_registers_pydantic_model_when_available() -> None:
    registry = HasherRegistry()
    _register_default_hashers(registry)
    assert isinstance(registry.find_hasher(pydantic.BaseModel), PydanticModelHasher)


def test_register_default_hashers_skips_pydantic_model_when_unavailable() -> None:
    registry = HasherRegistry()
    with (
        patch("coola.hashing.interface.is_pydantic_available", lambda: False),
        patch("coola.hashing.interface.PydanticModelHasher") as hasher_cls,
    ):
        _register_default_hashers(registry)
    hasher_cls.assert_not_called()
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
        result = hash_object([1], registry)
    assert result == hash_object([1], registry=registry)


def test_hash_object_positional_registry_none_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="positional arguments is deprecated"):
        result = hash_object([1], None)
    assert result == hash_object([1])


def test_hash_object_positional_length_and_ignore_unhashable_are_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="positional arguments is deprecated"):
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


class _HasHash:
    def hash(
        self,
        registry: object = None,  # noqa: ARG002
        length: int = 64,
        ignore_unhashable: bool = False,  # noqa: ARG002
    ) -> str:
        return "a" * length


def test_hash_object_dataclass() -> None:
    assert hash_object(_Point(1, 2)) == hash_object(_Point(1, 2))
    assert hash_object(_Point(1, 2)) != hash_object(_Point(2, 1))


def test_hash_object_nested_dataclass() -> None:
    line = _Line(_Point(0, 0), _Point(1, 1))
    assert hash_object(line) == hash_object(_Line(_Point(0, 0), _Point(1, 1)))
    assert hash_object(line) != hash_object(_Line(_Point(0, 0), _Point(1, 2)))
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


def test_hash_object_dataclass_registered_hasher_wins() -> None:
    registry = HasherRegistry()
    _register_default_hashers(registry)
    hasher = StrHasher()
    registry.register(_Line, hasher)
    assert registry.find_hasher(_Line) is hasher
    assert isinstance(registry.find_hasher(_Point), DataclassHasher)


def test_hash_object_supports_hash() -> None:
    assert hash_object(_HasHash(), length=8) != hash_object(
        _HasHash(), length=8, registry=HasherRegistry({object: StrHasher()})
    )
    assert hash_object(_HasHash()) == hash_object(_HasHash())


def test_hash_object_dataclass_with_hash_method_uses_the_dataclass_hasher() -> None:
    @dataclass
    class Both:
        x: int

        def hash(
            self,
            registry: object = None,  # noqa: ARG002
            length: int = 64,
            ignore_unhashable: bool = False,  # noqa: ARG002
        ) -> str:
            return "b" * length

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
def test_hash_object_standard_library_values(data: object) -> None:
    assert len(hash_object(data)) == 64
    assert hash_object(data) == hash_object(data)


def test_hash_object_standard_library_values_are_different() -> None:
    assert hash_object(UUID(int=1)) != hash_object(UUID(int=2))
    assert hash_object(UUID(int=1)) != hash_object(str(UUID(int=1)))
    assert hash_object(ZoneInfo("UTC")) != hash_object(ZoneInfo("Europe/Paris"))
    assert hash_object(Decimal(1)) != hash_object(1)
    assert hash_object(Fraction(1, 1)) != hash_object(1)
    assert hash_object(range(3)) != hash_object([0, 1, 2])


def test_hash_object_equal_values_have_the_same_hash() -> None:
    assert hash_object(Decimal("1.0")) == hash_object(Decimal("1.00"))
    assert hash_object(Fraction(1, 2)) == hash_object(Fraction(2, 4))
    assert hash_object(range(0)) == hash_object(range(5, 5))
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
    import numpy as np

    assert hash_object(getattr(np, scalar)(builtin)) == hash_object(builtin)


@numpy_available
def test_hash_object_numpy_scalar_in_a_container() -> None:
    import numpy as np

    assert hash_object({"a": [np.float32(0.5), np.int64(2)]}) == hash_object({"a": [0.5, 2]})


@numpy_available
def test_hash_object_numpy_scalar_different_values() -> None:
    import numpy as np

    assert hash_object(np.float32(1.0)) != hash_object(np.float32(2.0))
    assert hash_object(np.int64(1)) != hash_object(np.float64(1.0))
    assert hash_object(np.bool_(True)) != hash_object(1)


def test_hash_object_hash_method_with_another_signature_is_not_supports_hash() -> None:
    class Other:
        def hash(self, seed: int = 0) -> int:
            return seed

    with pytest.raises(TypeNotRegisteredError, match="register a hasher"):
        hash_object(Other())
    with pytest.warns(UserWarning, match="No hasher registered"):
        hash_object(Other(), ignore_unhashable=True)


@polars_available
def test_hash_object_polars_series_uses_the_polars_hasher() -> None:
    import polars as pl

    assert hash_object(pl.Series([1, 2])) == hash_object(pl.Series([1, 2]))

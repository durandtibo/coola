from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from coola.hashing import (
    BytesHasher,
    DatetimeHasher,
    HasherRegistry,
    MappingHasher,
    PathHasher,
    ReprHasher,
    SequenceHasher,
    StringHasher,
    get_default_registry,
    hash_object,
    hash_tagged,
    interface as hashing_interface,
    register_hashers,
)
from coola.hashing.interface import _register_default_hashers
from coola.hashing.pydantic import PydanticModelHasher
from coola.registry import AlreadyRegisteredError
from coola.testing.fixtures import pydantic_available
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

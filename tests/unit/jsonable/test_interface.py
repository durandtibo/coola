from __future__ import annotations

import datetime
import json
import warnings
from collections import ChainMap, Counter, OrderedDict, UserString, defaultdict, deque, namedtuple
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from uuid import UUID

import pytest

from coola.jsonable import (
    ConvertTransformer,
    DataclassTransformer,
    SequenceTransformer,
    get_default_registry,
    interface as jsonable_interface,
    register_transformers,
    to_jsonable,
)
from coola.recursive import DefaultTransformer, TransformerRegistry
from coola.registry import AlreadyRegisteredError
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if TYPE_CHECKING:
    from collections.abc import Generator

if is_numpy_available():
    import numpy as np


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    jsonable_interface._default_registry.reset()
    yield
    jsonable_interface._default_registry.reset()


class Obj:
    pass


class Color(IntEnum):
    RED = 1


class Name(StrEnum):
    A = "a"


class Plain(Enum):
    TUPLE = (1, PurePosixPath("p"))


@dataclass
class Point:
    x: Any
    y: Any


Pair = namedtuple("Pair", ["a", "b"])  # noqa: PYI024


#####################################
#     Tests for to_jsonable         #
#####################################


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(1, 1, id="int"),
        pytest.param(1.5, 1.5, id="float"),
        pytest.param(True, True, id="bool"),
        pytest.param(None, None, id="none"),
        pytest.param("abc", "abc", id="str"),
        pytest.param([1, [2]], [1, [2]], id="list"),
        pytest.param((1, 2), [1, 2], id="tuple"),
        pytest.param(Pair(1, 2), [1, 2], id="namedtuple"),
        pytest.param(deque([1, 2]), [1, 2], id="deque"),
        pytest.param(range(3), [0, 1, 2], id="range"),
        pytest.param({3, 1, 2}, [1, 2, 3], id="set"),
        pytest.param(frozenset({"b", "a"}), ["a", "b"], id="frozenset"),
        pytest.param({"a": 1}, {"a": 1}, id="dict"),
        pytest.param(OrderedDict(a=1), {"a": 1}, id="ordered_dict"),
        pytest.param(defaultdict(int, a=1), {"a": 1}, id="defaultdict"),
        pytest.param(Counter("aab"), {"a": 2, "b": 1}, id="counter"),
        pytest.param(MappingProxyType({"a": 1}), {"a": 1}, id="mapping_proxy"),
        pytest.param(ChainMap({"a": 1}), {"a": 1}, id="chain_map"),
        pytest.param(Point(1, 2), {"x": 1, "y": 2}, id="dataclass"),
        pytest.param(UserString("ab"), "ab", id="user_string"),
        pytest.param(PurePosixPath("a/b"), "a/b", id="path"),
        pytest.param(UUID(int=1), "00000000-0000-0000-0000-000000000001", id="uuid"),
        pytest.param(Decimal("1.5"), "1.5", id="decimal"),
        pytest.param(b"a", "YQ==", id="bytes"),
        pytest.param(bytearray(b"a"), "YQ==", id="bytearray"),
        pytest.param(datetime.date(2020, 1, 2), "2020-01-02", id="date"),
        pytest.param(
            datetime.datetime(2020, 1, 2, 3, 4, tzinfo=datetime.UTC),
            "2020-01-02T03:04:00+00:00",
            id="datetime",
        ),
        pytest.param(datetime.time(3, 4), "03:04:00", id="time"),
        pytest.param(datetime.timedelta(seconds=90), 90.0, id="timedelta"),
        pytest.param(Plain.TUPLE, [1, "p"], id="enum_tuple_value"),
    ],
)
def test_to_jsonable(data: Any, expected: Any) -> None:
    assert to_jsonable(data) == expected


def test_to_jsonable_keeps_an_int_enum_member() -> None:
    assert type(to_jsonable(Color.RED)) is Color


def test_to_jsonable_keeps_a_str_enum_member() -> None:
    assert type(to_jsonable(Name.A)) is Name


def test_to_jsonable_nan_is_kept() -> None:
    assert repr(to_jsonable([float("nan")])) == "[nan]"


def test_to_jsonable_whole_output() -> None:
    data = {
        "p": Point((1, 2), {3, 1}),
        "paths": [PurePosixPath("a")],
        1: Pair(None, b"a"),
        PurePosixPath("k"): Plain.TUPLE,
    }
    assert to_jsonable(data) == {
        "p": {"x": [1, 2], "y": [1, 3]},
        "paths": ["a"],
        "1": [None, "YQ=="],
        "k": [1, "p"],
    }


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        pytest.param(PurePosixPath("a"), "a", id="path"),
        pytest.param(True, "true", id="bool"),
        pytest.param(1, "1", id="int"),
        pytest.param(1.5, "1.5", id="float"),
        pytest.param(None, "null", id="none"),
        pytest.param(Color.RED, "1", id="int_enum"),
        pytest.param(UUID(int=1), "00000000-0000-0000-0000-000000000001", id="uuid"),
    ],
)
def test_to_jsonable_key(key: Any, expected: str) -> None:
    assert to_jsonable({key: 0}) == {expected: 0}


@pytest.mark.parametrize("mode", ["raise", "warn", "keep"])
def test_to_jsonable_tuple_key_raises(mode: str) -> None:
    with pytest.raises(TypeError, match=r"a key of type list"):
        to_jsonable({(1, 2): 0}, on_unsupported=mode)


def test_to_jsonable_unsupported_key_raises() -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type Obj"):
        to_jsonable({Obj(): 0})


def test_to_jsonable_unsupported_key_keep_raises() -> None:
    with pytest.raises(TypeError, match=r"a key of type Obj"):
        to_jsonable({Obj(): 0}, on_unsupported="keep")


def test_to_jsonable_unsupported_key_warn_raises() -> None:
    with (
        pytest.raises(TypeError, match=r"a key of type Obj"),
        pytest.warns(UserWarning, match=r"kept as is"),
    ):
        to_jsonable({Obj(): 0}, on_unsupported="warn")


def test_to_jsonable_duplicate_keys_raises() -> None:
    with pytest.raises(ValueError, match=r"Two keys give the same JSON key: '1'"):
        to_jsonable({1: "a", "1": "b"})


def test_to_jsonable_cycle_raises() -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match=r"Cycle detected"):
        to_jsonable(data)


@pytest.mark.parametrize("data", [Obj(), 1 + 2j, len, Obj, Point, slice(1)])
def test_to_jsonable_unsupported_raises(data: Any) -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type .* to JSON"):
        to_jsonable(data)


def test_to_jsonable_unsupported_names_the_type_and_the_remedy() -> None:
    with pytest.raises(TypeError, match=r"type Obj to JSON: register a ConvertTransformer"):
        to_jsonable([Obj()])


def test_to_jsonable_unsupported_warn() -> None:
    obj = Obj()
    with pytest.warns(UserWarning, match=r"Cannot convert an object of type Obj to JSON"):
        result = to_jsonable([obj], on_unsupported="warn")
    assert result[0] is obj


def test_to_jsonable_supported_warn_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert to_jsonable([1, "a", None], on_unsupported="warn") == [1, "a", None]


def test_to_jsonable_unsupported_keep() -> None:
    obj = Obj()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert to_jsonable([obj], on_unsupported="keep")[0] is obj


def test_to_jsonable_invalid_on_unsupported_raises() -> None:
    with pytest.raises(ValueError, match=r"on_unsupported"):
        to_jsonable(1, on_unsupported="ignore")


def test_to_jsonable_invalid_on_unsupported_raises_before_any_work() -> None:
    with pytest.raises(ValueError, match=r"on_unsupported"):
        to_jsonable(Obj(), on_unsupported="ignore")


def test_to_jsonable_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match=r"positional"):
        to_jsonable(1, None)  # ty: ignore[too-many-positional-arguments]


def test_to_jsonable_does_not_modify_the_input() -> None:
    data = {"a": [1, (2, 3)], "s": {1, 2}}
    to_jsonable(data)
    assert data == {"a": [1, (2, 3)], "s": {1, 2}}


def test_to_jsonable_output_shares_no_container_with_the_input() -> None:
    data = [[1]]
    assert to_jsonable(data)[0] is not data[0]


def test_to_jsonable_is_idempotent() -> None:
    data = {"a": (1, {2, 3}), 1: PurePosixPath("p")}
    assert to_jsonable(to_jsonable(data)) == to_jsonable(data)


def test_to_jsonable_output_is_accepted_by_json() -> None:
    data = {"a": (1, {2, 3}), 1: PurePosixPath("p"), "d": datetime.date(2020, 1, 1)}
    assert json.loads(json.dumps(to_jsonable(data))) == to_jsonable(data)


def test_to_jsonable_custom_registry() -> None:
    registry = get_default_registry().copy()
    registry.register(Obj, ConvertTransformer(lambda _: "obj"))
    assert to_jsonable([Obj()], registry=registry) == ["obj"]


def test_to_jsonable_custom_registry_does_not_modify_the_default_one() -> None:
    registry = get_default_registry().copy()
    registry.register(Obj, ConvertTransformer(lambda _: "obj"))
    to_jsonable([Obj()], registry=registry)
    assert not get_default_registry().has_transformer(Obj)


def test_to_jsonable_does_not_modify_the_default_registry() -> None:
    before = list(get_default_registry().keys())
    to_jsonable([Point(1, 2), {1: (1,)}, {1}])
    assert list(get_default_registry().keys()) == before


def test_to_jsonable_is_deterministic_across_calls() -> None:
    data = {"s": {"b", "a", "c"}}
    assert to_jsonable(data) == to_jsonable(data)


###############################
#     Tests for the leaves    #
###############################


def test_keep() -> None:
    obj = Obj()
    assert jsonable_interface._keep(obj) is obj


@pytest.mark.parametrize("value", [1, 1.5, True, None, "a", Color.RED])
def test_raise_keeps_a_scalar(value: Any) -> None:
    assert jsonable_interface._raise(value) is value


def test_raise_rejects_an_object() -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type Obj"):
        jsonable_interface._raise(Obj())


@pytest.mark.parametrize("value", [1, 1.5, True, None, "a"])
def test_warn_keeps_a_scalar_silently(value: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert jsonable_interface._warn(value) is value


def test_warn_warns_for_an_object() -> None:
    obj = Obj()
    with pytest.warns(UserWarning, match=r"Cannot convert an object of type Obj"):
        assert jsonable_interface._warn(obj) is obj


#####################################
#     Tests for the registry        #
#####################################


def test_get_default_registry_is_a_singleton() -> None:
    assert get_default_registry() is get_default_registry()


def test_get_default_registry_is_a_transformer_registry() -> None:
    assert isinstance(get_default_registry(), TransformerRegistry)


def test_get_default_registry_object_is_a_leaf() -> None:
    assert isinstance(get_default_registry().find_transformer(Obj), DefaultTransformer)


def test_get_default_registry_dataclass_fallback() -> None:
    assert isinstance(get_default_registry().find_transformer(Point), DataclassTransformer)


def test_get_default_registry_dataclass_class_is_a_leaf() -> None:
    assert get_default_registry().find_transformer(type(Point)).expand(Point) is None


def test_get_default_registry_subclass_of_a_scalar_uses_its_base() -> None:
    class MyFloat(float):
        pass

    assert to_jsonable(MyFloat(1.5)) == 1.5


def test_get_default_registry_user_string_is_not_a_sequence() -> None:
    assert not isinstance(get_default_registry().find_transformer(UserString), SequenceTransformer)


def test_register_transformers() -> None:
    register_transformers({Obj: ConvertTransformer(lambda _: "obj")})
    assert to_jsonable({"a": Obj()}) == {"a": "obj"}


def test_register_transformers_existing_type_raises() -> None:
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        register_transformers({list: SequenceTransformer()})


def test_register_transformers_exist_ok() -> None:
    register_transformers({tuple: ConvertTransformer(len)}, exist_ok=True)
    assert to_jsonable((1, 2)) == 2


def test_register_transformers_a_more_specific_type_wins() -> None:
    class MyList(list):
        pass

    register_transformers({MyList: ConvertTransformer(lambda _: "mine")})
    assert to_jsonable([MyList(), []]) == ["mine", []]


@numpy_available
@patch("coola.jsonable.interface.package_available", lambda _: False)
def test_register_default_transformers_without_optional_package() -> None:
    registry = TransformerRegistry()
    jsonable_interface._register_default_transformers(registry)
    assert not registry.has(np.ndarray)


@numpy_available
def test_register_default_transformers_with_optional_package() -> None:
    registry = TransformerRegistry()
    jsonable_interface._register_default_transformers(registry)
    assert registry.has(np.ndarray)

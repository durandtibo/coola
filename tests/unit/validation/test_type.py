from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType

import pytest

from coola.validation import (
    validate_bool,
    validate_dict,
    validate_float,
    validate_frozenset,
    validate_int,
    validate_isinstance,
    validate_list,
    validate_mapping,
    validate_set,
    validate_str,
    validate_tuple,
)


def test_validate_isinstance_valid() -> None:
    validate_isinstance(1, int)


def test_validate_isinstance_valid_tuple() -> None:
    validate_isinstance(1, (int, float))


def test_validate_isinstance_invalid() -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'int'>"):
        validate_isinstance("abc", int)


def test_validate_isinstance_invalid_tuple() -> None:
    with pytest.raises(TypeError, match="value must be an instance of"):
        validate_isinstance("abc", (int, float))


def test_validate_isinstance_custom_name() -> None:
    with pytest.raises(TypeError, match="count must be an instance of <class 'int'>"):
        validate_isinstance("abc", int, name="count")


@pytest.mark.parametrize("value", [True, False])
def test_validate_bool_valid(value: bool) -> None:
    validate_bool(value)


@pytest.mark.parametrize("value", [0, 1, "True", None, 1.0])
def test_validate_bool_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'bool'>"):
        validate_bool(value)


def test_validate_bool_custom_name() -> None:
    with pytest.raises(TypeError, match="flag must be an instance of <class 'bool'>"):
        validate_bool(1, name="flag")


@pytest.mark.parametrize("value", [0, 1, -5])
def test_validate_int_valid(value: object) -> None:
    validate_int(value)


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_validate_int_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an int"):
        validate_int(value)


def test_validate_int_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be an int"):
        validate_int(object(), name="foo")


@pytest.mark.parametrize("value", [0.0, 1.5, float("nan")])
def test_validate_float_valid(value: object) -> None:
    validate_float(value)


@pytest.mark.parametrize("value", [1, True, "1.0", None])
def test_validate_float_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'float'>"):
        validate_float(value)


def test_validate_float_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be an instance of <class 'float'>"):
        validate_float(object(), name="foo")


@pytest.mark.parametrize("value", ["", "abc"])
def test_validate_str_valid(value: object) -> None:
    validate_str(value)


@pytest.mark.parametrize("value", [1, b"abc", None, ["a"]])
def test_validate_str_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'str'>"):
        validate_str(value)


def test_validate_str_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be an instance of <class 'str'>"):
        validate_str(object(), name="foo")


@pytest.mark.parametrize("value", [[], [1, 2]])
def test_validate_list_valid(value: object) -> None:
    validate_list(value)


@pytest.mark.parametrize("value", [(1, 2), {1}, "ab", None])
def test_validate_list_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'list'>"):
        validate_list(value)


def test_validate_list_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be an instance of <class 'list'>"):
        validate_list(object(), name="foo")


@pytest.mark.parametrize("value", [(), (1, 2)])
def test_validate_tuple_valid(value: object) -> None:
    validate_tuple(value)


@pytest.mark.parametrize("value", [[1, 2], {1}, "ab", None])
def test_validate_tuple_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'tuple'>"):
        validate_tuple(value)


def test_validate_tuple_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be an instance of <class 'tuple'>"):
        validate_tuple(object(), name="foo")


@pytest.mark.parametrize("value", [{}, {"a": 1}])
def test_validate_dict_valid(value: object) -> None:
    validate_dict(value)


@pytest.mark.parametrize("value", [[("a", 1)], {1}, "ab", None])
def test_validate_dict_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'dict'>"):
        validate_dict(value)


def test_validate_dict_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be an instance of <class 'dict'>"):
        validate_dict(object(), name="foo")


@pytest.mark.parametrize("value", [set(), {1, 2}])
def test_validate_set_valid(value: object) -> None:
    validate_set(value)


@pytest.mark.parametrize("value", [frozenset({1}), [1], {"a": 1}, None])
def test_validate_set_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'set'>"):
        validate_set(value)


def test_validate_set_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be an instance of <class 'set'>"):
        validate_set(object(), name="foo")


@pytest.mark.parametrize("value", [frozenset(), frozenset({1})])
def test_validate_frozenset_valid(value: object) -> None:
    validate_frozenset(value)


@pytest.mark.parametrize("value", [{1}, [1], (1,), None])
def test_validate_frozenset_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of <class 'frozenset'>"):
        validate_frozenset(value)


def test_validate_frozenset_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be an instance of <class 'frozenset'>"):
        validate_frozenset(object(), name="foo")


@pytest.mark.parametrize("value", [{}, {"a": 1}, OrderedDict(a=1), MappingProxyType({"a": 1})])
def test_validate_mapping_valid(value: object) -> None:
    validate_mapping(value)


@pytest.mark.parametrize("value", [[("a", 1)], {1}, "ab", None])
def test_validate_mapping_invalid(value: object) -> None:
    with pytest.raises(TypeError, match=r"value must be an instance of .*Mapping"):
        validate_mapping(value)


def test_validate_mapping_custom_name() -> None:
    with pytest.raises(TypeError, match=r"foo must be an instance of .*Mapping"):
        validate_mapping(object(), name="foo")

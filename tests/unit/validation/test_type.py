from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from coola.validation import validate_int, validate_isinstance, validate_number


def test_validate_isinstance_valid() -> None:
    validate_isinstance(1, int)


def test_validate_isinstance_valid_tuple() -> None:
    validate_isinstance(1, (int, float))


def test_validate_isinstance_invalid() -> None:
    with pytest.raises(TypeError, match="value must be an instance of int"):
        validate_isinstance("abc", int)


def test_validate_isinstance_invalid_tuple() -> None:
    with pytest.raises(TypeError, match="value must be an instance of"):
        validate_isinstance("abc", (int, float))


def test_validate_isinstance_custom_name() -> None:
    with pytest.raises(TypeError, match="count must be an instance of int"):
        validate_isinstance("abc", int, name="count")


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


def test_validate_isinstance_tuple_message() -> None:
    with pytest.raises(TypeError, match=r"^value must be an instance of int or str, got float$"):
        validate_isinstance(1.5, (int, str))


def test_validate_isinstance_union_message() -> None:
    with pytest.raises(TypeError, match=r"^value must be an instance of int or str, got float$"):
        validate_isinstance(1.5, int | str)


def test_validate_isinstance_union_valid() -> None:
    validate_isinstance("a", int | str)


def test_validate_int_message_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"^count must be an int, got bool$"):
        validate_int(True, name="count")


@pytest.mark.parametrize("value", [{}, {"a": 1}, OrderedDict(a=1), MappingProxyType({"a": 1})])
def test_validate_isinstance_mapping_valid(value: object) -> None:
    validate_isinstance(value, Mapping)


@pytest.mark.parametrize("value", [[("a", 1)], {1}, "ab", None])
def test_validate_isinstance_mapping_invalid(value: object) -> None:
    with pytest.raises(TypeError, match="value must be an instance of Mapping, got"):
        validate_isinstance(value, Mapping)


def test_validate_isinstance_accepts_bool_as_int() -> None:
    validate_isinstance(True, int)


@pytest.mark.parametrize("value", [0, 1, -1.5, 1e300, float("inf")])
def test_validate_number_valid(value: float) -> None:
    validate_number(value)


@pytest.mark.parametrize("value", [True, False, None, "a", [1], 1j])
def test_validate_number_invalid(value: Any) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        validate_number(value)


def test_validate_number_custom_name() -> None:
    with pytest.raises(TypeError, match="foo must be a real number"):
        validate_number("a", name="foo")


def test_validate_number_message_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"^rtol must be a real number, got NoneType$"):
        validate_number(None, name="rtol")

from __future__ import annotations

import pytest

from coola.validation import validate_bool, validate_isinstance


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
    with pytest.raises(TypeError, match="value must be a bool"):
        validate_bool(value)


def test_validate_bool_custom_name() -> None:
    with pytest.raises(TypeError, match="flag must be a bool"):
        validate_bool(1, name="flag")

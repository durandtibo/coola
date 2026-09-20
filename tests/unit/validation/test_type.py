from __future__ import annotations

import pytest

from coola.validation import validate_isinstance


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

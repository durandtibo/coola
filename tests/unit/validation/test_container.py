from __future__ import annotations

from typing import Any

import pytest

from coola.validation import validate_in, validate_not_empty

########################################
#     Tests for validate_not_empty     #
########################################


@pytest.mark.parametrize("value", [[1, 2, 3], (1,), "abc", {1, 2}, {"a": 1}])
def test_validate_not_empty_valid(value: Any) -> None:
    validate_not_empty(value)


@pytest.mark.parametrize("value", [[], (), "", set(), {}])
def test_validate_not_empty_empty(value: Any) -> None:
    with pytest.raises(ValueError, match="value must not be empty"):
        validate_not_empty(value)


def test_validate_not_empty_custom_name() -> None:
    with pytest.raises(ValueError, match="my_list must not be empty"):
        validate_not_empty([], name="my_list")


##################################
#     Tests for validate_in     #
##################################


def test_validate_in_valid() -> None:
    validate_in("a", ("a", "b", "c"))


def test_validate_in_invalid() -> None:
    with pytest.raises(ValueError, match=r"value must be one of \('a', 'b', 'c'\), got 'd'"):
        validate_in("d", ("a", "b", "c"))


def test_validate_in_custom_name() -> None:
    with pytest.raises(ValueError, match=r"mode must be one of \('a', 'b'\), got 'c'"):
        validate_in("c", ("a", "b"), name="mode")

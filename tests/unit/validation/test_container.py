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


#################################
#     Tests for validate_in     #
#################################


def test_validate_in_valid() -> None:
    validate_in("a", ("a", "b", "c"))


def test_validate_in_invalid() -> None:
    with pytest.raises(ValueError, match=r"value must be one of \('a', 'b', 'c'\), got 'd'"):
        validate_in("d", ("a", "b", "c"))


def test_validate_in_custom_name() -> None:
    with pytest.raises(ValueError, match=r"mode must be one of \('a', 'b'\), got 'c'"):
        validate_in("c", ("a", "b"), name="mode")


def test_validate_not_empty_without_length_raises() -> None:
    with pytest.raises(TypeError, match=r"^value must have a length, got int$"):
        validate_not_empty(5)  # ty: ignore[invalid-argument-type]


def test_validate_not_empty_without_length_custom_name() -> None:
    with pytest.raises(TypeError, match=r"^items must have a length, got int$"):
        validate_not_empty(5, name="items")  # ty: ignore[invalid-argument-type]


def test_validate_not_empty_does_not_consume_a_generator() -> None:
    gen = (i for i in range(3))
    with pytest.raises(TypeError, match="must have a length, got generator"):
        validate_not_empty(gen)  # ty: ignore[invalid-argument-type]
    assert next(gen) == 0


def test_validate_in_unhashable_value_in_a_set_raises_value_error() -> None:
    with pytest.raises(ValueError, match=r"^value must be one of \{1\}, got \[\]$"):
        validate_in([], {1})


def test_validate_in_long_container_message_is_bounded() -> None:
    with pytest.raises(ValueError, match="value must be one of") as exc_info:
        validate_in(-1, list(range(100000)))
    assert len(str(exc_info.value)) < 200


def test_validate_in_long_container_message_listing() -> None:
    with pytest.raises(ValueError, match="value must be one of") as exc_info:
        validate_in(-1, tuple(range(100)))
    assert str(exc_info.value) == (
        "value must be one of (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, "
        "19, ...), got -1"
    )


def test_validate_in_not_a_container_raises_type_error() -> None:
    with pytest.raises(TypeError, match="not a container or iterable"):
        validate_in(1, 5)  # ty: ignore[invalid-argument-type]


def test_validate_not_empty_does_not_modify_the_list() -> None:
    value = [1, 2]
    validate_not_empty(value)
    assert value == [1, 2]


def test_validate_not_empty_does_not_modify_the_dict_on_failure() -> None:
    value: dict[str, int] = {}
    with pytest.raises(ValueError, match="must not be empty"):
        validate_not_empty(value)
    assert value == {}


def test_validate_in_does_not_modify_the_valid_values() -> None:
    valid = ["a", "b"]
    with pytest.raises(ValueError, match="must be one of"):
        validate_in("c", valid)
    assert valid == ["a", "b"]

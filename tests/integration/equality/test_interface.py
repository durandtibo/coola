from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.equality import objects_are_equal
from coola.equality.tester import (
    SequenceEqualityTester,
    interface as tester_interface,
    register_equality_testers,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    tester_interface._default_registry.reset()
    yield
    tester_interface._default_registry.reset()


class CustomList(list): ...


def test_object_are_equal_custom_type() -> None:
    register_equality_testers({CustomList: SequenceEqualityTester()})
    assert objects_are_equal(CustomList([1, 2, 3]), CustomList([1, 2, 3]))


def test_objects_are_equal_large_lists_that_differ_at_the_end_are_different() -> None:
    first = list(range(100_000))
    assert not objects_are_equal(first, [*first[:-1], -1])


def test_objects_are_equal_dict_values_equal() -> None:
    assert objects_are_equal({"a": 1, "b": [2.0]}.values(), {"a": 1, "b": [2.0]}.values())


def test_objects_are_equal_dict_values_nested_in_a_dict_equal() -> None:
    assert objects_are_equal({"x": {"a": 1}.values()}, {"x": {"a": 1}.values()})


def test_objects_are_equal_dict_values_different_value() -> None:
    assert not objects_are_equal({"a": 1}.values(), {"a": 2}.values())


def test_objects_are_equal_dict_values_different_order() -> None:
    assert not objects_are_equal({"a": 1, "b": 2}.values(), {"b": 2, "a": 1}.values())


def test_objects_are_equal_dict_values_different_length() -> None:
    assert not objects_are_equal({"a": 1}.values(), {"a": 1, "b": 2}.values())


def test_objects_are_equal_dict_values_nan_equal_nan() -> None:
    assert objects_are_equal(
        {"a": float("nan")}.values(), {"a": float("nan")}.values(), equal_nan=True
    )


def test_objects_are_equal_dict_values_and_list_are_different() -> None:
    assert not objects_are_equal({"a": 1}.values(), [1])

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.iterator import filter_by_type

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.mark.parametrize(
    ("data", "types", "expected"),
    [
        ([1, "hello", 2, 3.14, "world", 4], int, [1, 2, 4]),
        (iter([1, "hello", 2, 3.14, "world", 4]), int, [1, 2, 4]),
        ([1, "hello", 2.5, True, None, [1, 2]], (int, float), [1, 2.5, True]),
        (["apple", "banana", "cherry"], (int, float), []),
        ([], int, []),
        ([1, 2, 3, True, False, "hello"], int, [1, 2, 3, True, False]),
        ([1, "hello", 2.5, True, None, [1, 2]], (int, str, bool), [1, "hello", True]),
        ((1, "hello", 2, 3.14, "world", 4), int, [1, 2, 4]),
        ({"a": 1, "b": 2.5, "c": "hello"}.values(), (int, float), [1, 2.5]),
        (range(5), int, [0, 1, 2, 3, 4]),
    ],
)
def test_filter_by_type(
    data: Iterator[Any], types: type | tuple[type, ...], expected: Iterator[Any]
) -> None:
    assert list(filter_by_type(data, types)) == expected


def test_filter_by_type_iterable_input_generator() -> None:
    """Test that the function works with an iterable input (like a
    tuple)"""

    def gen() -> Iterator[Any]:
        yield 1
        yield 2
        yield "abc"
        yield 4

    assert list(filter_by_type(gen(), int)) == [1, 2, 4]


def test_filter_by_type_tuple_of_types_keeps_the_order_of_the_input() -> None:
    result: list[str | int] = list(filter_by_type(["b", 2, "a", 1], (int, str)))
    assert result == ["b", 2, "a", 1]


def test_filter_by_type_tuple_with_one_type() -> None:
    result: list[int] = list(filter_by_type([1, "a", 2], (int,)))
    assert result == [1, 2]

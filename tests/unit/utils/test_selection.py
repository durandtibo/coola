from __future__ import annotations

from typing import Any

import pytest

from coola.utils.selection import smallest_items, sort_items

####################################
#     Tests for smallest_items     #
####################################


def test_smallest_items_all_items_are_sorted() -> None:
    assert smallest_items({3, 1, 2}) == [1, 2, 3]


def test_smallest_items_limit() -> None:
    assert smallest_items({5, 3, 1, 4, 2}, limit=2) == [1, 2]


def test_smallest_items_limit_larger_than_the_items() -> None:
    assert smallest_items({2, 1}, limit=10) == [1, 2]


def test_smallest_items_limit_zero() -> None:
    assert smallest_items({2, 1}, limit=0) == []


def test_smallest_items_empty() -> None:
    assert smallest_items(set()) == []


def test_smallest_items_duplicate_items_are_kept() -> None:
    assert smallest_items([2, 1, 2], limit=3) == [1, 2, 2]


def test_smallest_items_strings() -> None:
    assert smallest_items({"b", "c", "a"}, limit=2) == ["a", "b"]


@pytest.mark.parametrize(("limit", "expected"), [(1, [1]), (2, [1, 2])])
def test_smallest_items_not_comparable_items_are_sorted_by_type_name(
    limit: int, expected: list
) -> None:
    assert smallest_items([2, "a", 1], limit=limit) == expected


def test_smallest_items_not_comparable_items_do_not_depend_on_the_order() -> None:
    assert smallest_items(["a", 1], limit=1) == smallest_items([1, "a"], limit=1)


def test_smallest_items_not_comparable_items_without_limit() -> None:
    assert smallest_items([1, "a"]) == [1, "a"]


def test_smallest_items_does_not_modify_the_data() -> None:
    data = [3, 1, 2]
    smallest_items(data, limit=2)
    assert data == [3, 1, 2]


def test_smallest_items_limit_is_keyword_only() -> None:
    function: Any = smallest_items
    with pytest.raises(TypeError, match="positional argument"):
        function([3, 1, 2], 2)


def test_smallest_items_negative_limit_raises() -> None:
    with pytest.raises(ValueError, match=r"limit must be greater than or equal to 0, got -1"):
        smallest_items([3, 1, 2], limit=-1)


###############################
#     Tests for sort_items     #
###############################


def test_sort_items_comparable_items() -> None:
    assert sort_items({3, 1, 2}) == [1, 2, 3]


def test_sort_items_empty() -> None:
    assert sort_items([]) == []


def test_sort_items_duplicate_items_are_kept() -> None:
    assert sort_items([2, 1, 2]) == [1, 2, 2]


def test_sort_items_accepts_an_iterator() -> None:
    assert sort_items(iter([2, 1])) == [1, 2]


def test_sort_items_not_comparable_items_are_sorted_by_type_name() -> None:
    assert sort_items([2, "b", 1, "a"]) == [1, 2, "a", "b"]


def test_sort_items_not_comparable_items_of_the_same_type_are_sorted_by_repr() -> None:
    assert sort_items([("b",), (1,), ("a",)]) == [("a",), ("b",), (1,)]


def test_sort_items_not_comparable_items_do_not_depend_on_the_order() -> None:
    assert sort_items([None, "a", 1]) == sort_items([1, None, "a"])


def test_sort_items_key() -> None:
    assert sort_items([("b", 1), ("a", 2)], key=lambda x: x[0]) == [("a", 2), ("b", 1)]


def test_sort_items_key_not_comparable_values() -> None:
    assert sort_items([("a", 1), (1, 2)], key=lambda x: x[0]) == [(1, 2), ("a", 1)]


def test_sort_items_does_not_modify_the_data() -> None:
    data = [3, 1, 2]
    sort_items(data)
    assert data == [3, 1, 2]


def test_sort_items_key_is_keyword_only() -> None:
    function: Any = sort_items
    with pytest.raises(TypeError, match="positional"):
        function([1], len)

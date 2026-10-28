from __future__ import annotations

import pytest

from coola.utils.selection import smallest_items

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


@pytest.mark.parametrize(("limit", "expected"), [(1, [1]), (2, [1, "a"])])
def test_smallest_items_not_comparable_items_keep_the_iteration_order(
    limit: int, expected: list
) -> None:
    assert smallest_items([1, "a", 2], limit=limit) == expected


def test_smallest_items_not_comparable_items_without_limit() -> None:
    assert smallest_items([1, "a"]) == [1, "a"]


def test_smallest_items_does_not_modify_the_data() -> None:
    data = [3, 1, 2]
    smallest_items(data, limit=2)
    assert data == [3, 1, 2]

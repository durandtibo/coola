from __future__ import annotations

from coola.utils.mapping import sort_by_keys, sort_by_values

##################################
#     Tests for sort_by_keys     #
##################################


def test_sort_by_keys_empty() -> None:
    assert sort_by_keys({}) == {}


def test_sort_by_keys() -> None:
    data = sort_by_keys({"dog": 1, "cat": 5, "fish": 2})
    assert data == {"cat": 5, "dog": 1, "fish": 2}
    assert list(data.keys()) == ["cat", "dog", "fish"]


####################################
#     Tests for sort_by_values     #
####################################


def test_sort_by_values_empty() -> None:
    assert sort_by_values({}) == {}


def test_sort_by_values() -> None:
    data = sort_by_values({"dog": 1, "cat": 5, "fish": 2})
    assert data == {"dog": 1, "fish": 2, "cat": 5}
    assert list(data.keys()) == ["dog", "fish", "cat"]
